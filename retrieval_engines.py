import sqlite3
import time
import math
import re
from collections import Counter

SEMANTIC_SYNONYMS = {
    "car": "automotive can bus wheel",
    "vehicle": "automotive can bus",
    "freeze": "deadlock locked stutter hang",
    "crash": "exits killed error fails crashloopbackoff",
    "memory": "oom ram dram cuda buffer heap",
    "ram": "memory oom dram",
    "slow": "latency stutter timeout drop lag",
    "lag": "stutter latency dropped frames offset",
    "android": "kotlin recyclerview mobile",
    "phone": "android kotlin mobile",
    "plane": "avionics arinc flight",
    "aircraft": "avionics arinc flight",
    "power": "grid scada transformer substation",
    "electricity": "grid scada transformer",
    "docker": "container oom pod",
    "gpu": "cuda pytorch machine learning",
    "skeleton": "biomechanics spine marker kinematics",
    "back": "spine biomechanics"
}

class RetrievalEngine:
    def __init__(self, db_path, dataset):
        self.db_path = db_path
        self.dataset = dataset
        self.doc_map = {row[0]: {"id": row[0], "code": row[1], "cat": row[2], "content": row[3]} for row in dataset}
        self.idf = {}
        self.doc_vectors = {}
        self._build_vector_index()

    @staticmethod
    def tokenize(text, expand_synonyms=False):
        words = re.findall(r'[a-z0-9_]+', text.lower())
        if expand_synonyms:
            expanded = list(words)
            for w in words:
                if w in SEMANTIC_SYNONYMS:
                    expanded.extend(SEMANTIC_SYNONYMS[w].split())
            return expanded
        return words

    def _build_vector_index(self):
        n_docs = len(self.dataset)
        df = Counter()
        doc_tokens = {}

        for row_id, code, cat, content in self.dataset:
            full_text = f"{code} {cat} {content}"
            tokens = self.tokenize(full_text)
            # Extract character 3-grams on key terms for typo resilience
            trigrams = [code.lower()[i:i+3] for i in range(max(0, len(code)-2))]
            for word in tokens[:18]:
                if len(word) >= 4:
                    trigrams.extend(word[i:i+3] for i in range(len(word)-2))
            combined = tokens + trigrams
            doc_tokens[row_id] = combined
            for term in set(combined):
                df[term] += 1

        for term, count in df.items():
            self.idf[term] = math.log((n_docs + 1) / (count + 1)) + 1.0

        for doc_id, terms in doc_tokens.items():
            tf = Counter(terms)
            total = len(terms) or 1
            vec = {t: (cnt / total) * self.idf.get(t, 1.0) for t, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors[doc_id] = {t: v / norm for t, v in vec.items()}

    def search_sql_like(self, query):
        t0 = time.perf_counter()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        words = [w for w in self.tokenize(query) if len(w) > 2] or [query.strip().lower()]

        results = []
        for row_id, code, cat, content in c.execute("SELECT id, code, category, content FROM kb_standard"):
            haystack = f"{code} {cat} {content}".lower()
            hits = sum(1 for w in words if w in haystack)
            if hits > 0:
                score = round((hits / len(words)) * 100.0, 1)
                results.append({"id": row_id, "code": code, "cat": cat, "content": content, "score": score})
        conn.close()
        results.sort(key=lambda x: (x["score"], -x["id"]), reverse=True)
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return results[:3], latency_ms

    def search_fts5_bm25(self, query):
        t0 = time.perf_counter()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        words = [w for w in self.tokenize(query) if len(w) > 1]
        fts_query = " OR ".join(f'"{w}"*' for w in words) if words else '""'

        results = []
        try:
            sql = "SELECT id, code, category, content, bm25(kb_fts) as rank FROM kb_fts WHERE kb_fts MATCH ? ORDER BY rank LIMIT 3"
            rows = c.execute(sql, (fts_query,)).fetchall()
            for row_id, code, cat, content, rank in rows:
                raw = abs(rank)
                # Normalized saturation curve: raw / (raw + k)
                norm_score = round((raw / (raw + 2.5)) * 100.0, 1)
                results.append({"id": row_id, "code": code, "cat": cat, "content": content, "score": norm_score})
        except Exception:
            pass
        conn.close()
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return results, latency_ms

    def search_vector_cosine(self, query):
        t0 = time.perf_counter()
        q_tokens = self.tokenize(query, expand_synonyms=True)
        q_trigrams = []
        for word in self.tokenize(query):
            if len(word) >= 4:
                q_trigrams.extend(word[i:i+3] for i in range(len(word)-2))

        all_q = q_tokens + q_trigrams
        tf = Counter(all_q)
        total = max(1, len(all_q))
        q_vec = {t: (cnt / total) * self.idf.get(t, 1.0) for t, cnt in tf.items() if t in self.idf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}

        scored = []
        for doc_id, d_vec in self.doc_vectors.items():
            cosine = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
            if cosine > 0.05:
                norm_score = round(min(99.4, math.sqrt(cosine) * 100.0), 1)
                item = dict(self.doc_map[doc_id])
                item["score"] = norm_score
                scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return scored[:3], latency_ms

    def search_hybrid_rrf(self, query, k=10):
        t0 = time.perf_counter()
        fts_res, _ = self.search_fts5_bm25(query)
        vec_res, _ = self.search_vector_cosine(query)

        rrf_scores = Counter()
        for rank, item in enumerate(fts_res):
            rrf_scores[item["id"]] += 1.0 / (k + rank + 1)
        for rank, item in enumerate(vec_res):
            rrf_scores[item["id"]] += 1.0 / (k + rank + 1)

        max_theoretical = 2.0 / (k + 1)
        results = []
        for doc_id, rrf_val in rrf_scores.most_common(3):
            item = dict(self.doc_map[doc_id])
            item["score"] = round(min(99.8, (rrf_val / max_theoretical) * 100.0), 1)
            results.append(item)

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return results, latency_ms
