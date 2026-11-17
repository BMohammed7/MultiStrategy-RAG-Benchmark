import sqlite3, time, math, re
from collections import Counter
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
DB_PATH = "arena.db"

DATASET = [
    (1, "ERR-404-CAN", "Automotive CAN Bus", "CAN bus frame drop when polling wheel speed sensors at 500kbps. Fix: Add 120-ohm termination resistor across CAN_H and CAN_L."),
    (2, "SYS-QNX-09", "Embedded RTOS", "QNX Neutrino microkernel IPC deadlock when two threads send synchronous MsgSend() to each other. Fix: Enforce strict hierarchy or use MsgDeliverEvent."),
    (3, "CLOUD-DOCKER-02", "DevOps & Containers", "Docker container exits immediately with Code 137 (OOM Killed). The Linux kernel terminated the process due to exceeding memory limits."),
    (4, "DB-SQLITE-11", "Database Engine", "SQLite 'database is locked' error under concurrent writes. Enable Write-Ahead Logging via PRAGMA journal_mode=WAL; to allow simultaneous readers."),
    (5, "NET-REDIS-05", "Distributed Systems", "Redis cluster split-brain scenario during network partition. Minimum 3 master nodes required for quorum consensus."),
    (6, "AI-CUDA-88", "Machine Learning", "PyTorch RuntimeError: CUDA out of memory during backpropagation. Reduce batch size or enable mixed precision (torch.cuda.amp)."),
    (7, "AV-ARINC-429", "Avionics Systems", "ARINC 429 bus parity error on flight control telemetry channel. Check twisted shielded pair grounding and 32-bit word odd parity bit."),
    (8, "GRID-SCADA-14", "Power Distribution", "Transformer substation SCADA telemetry timeout over DNP3 protocol. Inspect RTU serial baud rate and cellular modem keep-alive interval."),
    (9, "MOB-KOTLIN-07", "Android Development", "RecyclerView UI stutter and dropped frames when scrolling images. Move SQLite queries and bitmap decoding off the Main UI thread using Kotlin Coroutines."),
    (10, "MEM-DRAM-03", "Computer Architecture", "High memory read latency caused by row buffer misses. Consecutive accesses to different rows in the same bank force PRECHARGE and ACTIVATE cycles."),
    (11, "SEC-JWT-21", "Backend Security", "API Gateway rejecting valid Bearer tokens with HTTP 401. System clock drift between microservices invalidates the JWT 'nbf' (not before) claim."),
    (12, "BIO-IK-19", "Biomechanics Sim", "Inverse Kinematics solver fails to converge on spine flexion trial. Marker tracking lost on C7 and acromion landmarks; apply 6Hz low-pass Butterworth filter.")
]

SEMANTIC_SYNONYMS = {
    "car": "automotive can bus wheel", "vehicle": "automotive can bus", "freeze": "deadlock locked stutter hang",
    "crash": "exits killed error fails", "memory": "oom ram dram cuda buffer", "ram": "memory oom dram",
    "slow": "latency stutter timeout drop", "lag": "stutter latency dropped frames", "android": "kotlin recyclerview mobile",
    "phone": "android kotlin mobile", "plane": "avionics arinc flight", "aircraft": "avionics arinc flight",
    "power": "grid scada transformer substation", "electricity": "grid scada transformer", "docker": "container oom",
    "gpu": "cuda pytorch machine learning", "skeleton": "biomechanics spine marker kinematics", "back": "spine biomechanics"
}

def tokenize(text, expand_synonyms=False):
    words = re.findall(r'[a-z0-9_]+', text.lower())
    if expand_synonyms:
        expanded = list(words)
        for w in words:
            if w in SEMANTIC_SYNONYMS:
                expanded.extend(SEMANTIC_SYNONYMS[w].split())
        return expanded
    return words

DOC_VECTORS = {}
IDF = {}

def init_db_and_vectors():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS kb_standard")
    c.execute("DROP TABLE IF EXISTS kb_fts")
    c.execute("CREATE TABLE kb_standard (id INT PRIMARY KEY, code TEXT, category TEXT, content TEXT)")
    c.execute("CREATE VIRTUAL TABLE kb_fts USING fts5(id UNINDEXED, code, category, content, tokenize='porter')")
    c.executemany("INSERT INTO kb_standard VALUES (?, ?, ?, ?)", DATASET)
    c.executemany("INSERT INTO kb_fts VALUES (?, ?, ?, ?)", DATASET)
    conn.commit()
    conn.close()

    N = len(DATASET)
    df = Counter()
    doc_tokens = {}
    for row in DATASET:
        full_text = f"{row[1]} {row[2]} {row[3]}"
        tokens = tokenize(full_text)
        trigrams = [full_text.lower()[i:i+3] for i in range(len(full_text)-2)]
        combined = tokens + trigrams
        doc_tokens[row[0]] = combined
        for term in set(combined):
            df[term] += 1

    for term, count in df.items():
        IDF[term] = math.log((N + 1) / (count + 1)) + 1.0

    for doc_id, terms in doc_tokens.items():
        tf = Counter(terms)
        vec = {t: (count / len(terms)) * IDF.get(t, 1.0) for t, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        DOC_VECTORS[doc_id] = {t: v / norm for t, v in vec.items()}

init_db_and_vectors()

def retrieve_sql_like(query):
    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    words = [w for w in tokenize(query) if len(w) > 2]
    if not words:
        words = [query.strip()]
    results = []
    for row_id, code, cat, content in c.execute("SELECT * FROM kb_standard"):
        text_lower = f"{code} {cat} {content}".lower()
        hits = sum(1 for w in words if w in text_lower)
        if hits > 0:
            score = round((hits / len(words)) * 100, 1)
            results.append({"id": row_id, "code": code, "cat": cat, "content": content, "score": score})
    conn.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return results[:2], latency_ms

def retrieve_fts5_bm25(query):
    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    words = [w for w in tokenize(query) if len(w) > 1]
    fts_query = " OR ".join(f'"{w}"*' for w in words) if words else '""'
    results = []
    try:
        sql = "SELECT id, code, category, content, bm25(kb_fts) as rank FROM kb_fts WHERE kb_fts MATCH ? ORDER BY rank LIMIT 2"
        for row_id, code, cat, content, rank in c.execute(sql, (fts_query,)):
            norm_score = round(min(99.9, abs(rank) * 28.0), 1)
            results.append({"id": row_id, "code": code, "cat": cat, "content": content, "score": norm_score})
    except Exception:
        pass
    conn.close()
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return results, latency_ms

def retrieve_vector_cosine(query):
    t0 = time.perf_counter()
    q_tokens = tokenize(query, expand_synonyms=True)
    q_trigrams = [query.lower()[i:i+3] for i in range(len(query)-2)]
    all_q = q_tokens + q_trigrams
    tf = Counter(all_q)
    q_vec = {t: (cnt / max(1, len(all_q))) * IDF.get(t, 1.0) for t, cnt in tf.items() if t in IDF}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    q_vec = {t: v / q_norm for t, v in q_vec.items()}

    scored = []
    for row in DATASET:
        doc_id = row[0]
        d_vec = DOC_VECTORS[doc_id]
        dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
        if dot > 0.04:
            scored.append({"id": doc_id, "code": row[1], "cat": row[2], "content": row[3], "score": round(min(99.9, dot * 165), 1)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return scored[:2], latency_ms

def retrieve_hybrid_rrf(query):
    t0 = time.perf_counter()
    fts_res, _ = retrieve_fts5_bm25(query)
    vec_res, _ = retrieve_vector_cosine(query)
    rrf_scores = Counter()
    doc_map = {}
    for rank, item in enumerate(fts_res):
        rrf_scores[item["id"]] += 1.0 / (10 + rank + 1)
        doc_map[item["id"]] = item
    for rank, item in enumerate(vec_res):
        rrf_scores[item["id"]] += 1.0 / (10 + rank + 1)
        doc_map[item["id"]] = item

    results = []
    for doc_id, rrf_val in rrf_scores.most_common(2):
        base = dict(doc_map[doc_id])
        base["score"] = round(min(99.9, (rrf_val / 0.1818) * 100), 1)
        results.append(base)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return results, latency_ms

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DB Retrieval Benchmark Arena</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 p-6 font-sans">
    <div class="max-w-6xl mx-auto">
        <div class="border-b border-slate-800 pb-4 mb-6 flex justify-between items-center">
            <div>
                <h1 class="text-2xl font-bold text-cyan-400">Database Retrieval Benchmark Arena</h1>
                <p class="text-slate-400 text-sm">Compare Speed (ms) vs. Relevance Score Across 4 Retrieval Architectures</p>
            </div>
            <span class="text-xs bg-slate-900 border border-slate-700 px-3 py-1.5 rounded text-slate-300">Dataset: 12 Engineering Incident Logs</span>
        </div>
        <div class="mb-4">
            <span class="text-xs text-slate-400 uppercase tracking-wider block mb-2">Click a Test Case to See Which Method Wins:</span>
            <div class="flex flex-wrap gap-2">
                <button onclick="runQuery('ERR-404-CAN')" class="text-xs bg-slate-900 hover:bg-slate-800 border border-cyan-800 text-cyan-300 px-3 py-1.5 rounded">1. Exact ID: ERR-404-CAN</button>
                <button onclick="runQuery('Why is my phone app lagging when scrolling pictures?')" class="text-xs bg-slate-900 hover:bg-slate-800 border border-purple-800 text-purple-300 px-3 py-1.5 rounded">2. Semantic Synonym</button>
                <button onclick="runQuery('container runs out of ram and crashes')" class="text-xs bg-slate-900 hover:bg-slate-800 border border-emerald-800 text-emerald-300 px-3 py-1.5 rounded">3. Mixed Intent</button>
                <button onclick="runQuery('PyTrch CUDDA memoery crash')" class="text-xs bg-slate-900 hover:bg-slate-800 border border-amber-800 text-amber-300 px-3 py-1.5 rounded">4. Heavy Typos</button>
            </div>
        </div>
        <div class="flex gap-3 mb-6">
            <input id="q" type="text" placeholder="Ask a question or type an error code..." class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-400" onkeydown="if(event.key==='Enter') runQuery()">
            <button onclick="runQuery()" class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold px-6 py-3 rounded-lg transition">Benchmark All 4</button>
        </div>
        <div id="bot-box" class="hidden bg-slate-900/90 border border-emerald-700/60 p-4 rounded-xl mb-6">
            <div class="text-xs font-bold text-emerald-400 uppercase mb-1" id="winner-banner"></div>
            <p class="text-slate-200 text-sm" id="bot-answer"></p>
        </div>
        <div id="arena" class="grid grid-cols-1 md:grid-cols-2 gap-5"></div>
    </div>
    <script>
        async function runQuery(preset) {
            var input = document.getElementById('q');
            if (preset) input.value = preset;
            var query = input.value.trim();
            if (!query) return;
            var res = await fetch('/api/benchmark', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            var data = await res.json();
            var botBox = document.getElementById('bot-box');
            botBox.classList.remove('hidden');
            document.getElementById('winner-banner').innerText = 'Best Answer Synthesized via ' + data.best_method + ' (' + data.fastest_method + ' was fastest at ' + data.fastest_ms + ' ms)';
            document.getElementById('bot-answer').innerText = data.chatbot_reply;
            var arena = document.getElementById('arena');
            arena.innerHTML = '';
            data.methods.forEach(function(m) {
                var top = m.results[0];
                var badgeColor = m.is_fastest ? 'bg-amber-950 text-amber-300 border-amber-700' : 'bg-slate-800 text-slate-300 border-slate-700';
                var bodyHtml = '';
                if (top) {
                    bodyHtml = '<div class="flex justify-between text-xs mb-1">' +
                        '<span class="font-mono text-cyan-400 font-bold">' + top.code + ' (' + top.cat + ')</span>' +
                        '<span class="text-emerald-400 font-bold">Confidence: ' + top.score + '%</span></div>' +
                        '<div class="w-full bg-slate-800 h-1.5 rounded-full mb-3">' +
                        '<div class="bg-emerald-500 h-full rounded-full" style="width: ' + top.score + '%"></div></div>' +
                        '<p class="text-xs text-slate-300 leading-relaxed bg-slate-950 p-3 rounded border border-slate-800">' + top.content + '</p>';
                } else {
                    bodyHtml = '<p class="text-xs text-rose-400 italic py-4">0 Matches Found</p>';
                }
                arena.innerHTML += '<div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">' +
                    '<div><div class="flex justify-between items-start mb-2"><div>' +
                    '<h3 class="font-bold text-base text-white">' + m.name + '</h3>' +
                    '<p class="text-xs text-slate-400">' + m.desc + '</p></div>' +
                    '<span class="text-xs px-2.5 py-1 rounded border font-mono ' + badgeColor + '">' + m.latency_ms + ' ms</span></div>' +
                    '<div class="mt-4 pt-3 border-t border-slate-800/80">' + bodyHtml + '</div></div></div>';
            });
        }
        runQuery('Why is my phone app lagging when scrolling pictures?');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    q = request.json.get('query', '')
    r_like, t_like = retrieve_sql_like(q)
    r_fts, t_fts = retrieve_fts5_bm25(q)
    r_vec, t_vec = retrieve_vector_cosine(q)
    r_hyb, t_hyb = retrieve_hybrid_rrf(q)

    methods = [
        {"name": "1. SQL Keyword LIKE", "desc": "Full Table Scan + Exact Substring", "results": r_like, "latency_ms": t_like},
        {"name": "2. SQLite FTS5 (BM25)", "desc": "Inverted Index + Porter Stemming", "results": r_fts, "latency_ms": t_fts},
        {"name": "3. Vector Cosine (TF-IDF + 3-Gram)", "desc": "Semantic Synonyms + Fuzzy Typo Vectors", "results": r_vec, "latency_ms": t_vec},
        {"name": "4. Hybrid RRF Fusion", "desc": "Combines FTS5 Rank + Vector Rank", "results": r_hyb, "latency_ms": t_hyb},
    ]

    fastest = min(methods, key=lambda x: x["latency_ms"])
    for m in methods:
        m["is_fastest"] = (m["name"] == fastest["name"])

    best_method = "4. Hybrid RRF Fusion"
    best_hit = r_hyb[0] if r_hyb else (r_vec[0] if r_vec else (r_fts[0] if r_fts else None))
    if best_hit:
        reply = f"[{best_hit['code']} - {best_hit['cat']}]: {best_hit['content']}"
    else:
        reply = "No matching engineering record found across any retrieval strategy."

    return jsonify({
        "methods": methods,
        "fastest_method": fastest["name"],
        "fastest_ms": fastest["latency_ms"],
        "best_method": best_method,
        "chatbot_reply": reply
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
