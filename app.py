from flask import Flask, render_template, request, jsonify
from seed_data import build_database
from retrieval_engines import RetrievalEngine

app = Flask(__name__)

DB_PATH = "arena.db"
DATASET = build_database(DB_PATH, target_rows=1200)
engine = RetrievalEngine(DB_PATH, DATASET)

@app.route("/")
def index():
    return render_template("index.html", total_rows=len(DATASET))

@app.route("/api/benchmark", methods=["POST"])
def benchmark():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query", "").strip()

    r_like, t_like = engine.search_sql_like(query)
    r_fts, t_fts = engine.search_fts5_bm25(query)
    r_vec, t_vec = engine.search_vector_cosine(query)
    r_hyb, t_hyb = engine.search_hybrid_rrf(query)

    methods = [
        {"name": "1. SQL Keyword LIKE", "desc": "Sequential Full-Table Scan (1,200 rows)", "results": r_like, "latency_ms": t_like},
        {"name": "2. SQLite FTS5 (BM25)", "desc": "B-Tree Inverted Index + Porter Stemmer", "results": r_fts, "latency_ms": t_fts},
        {"name": "3. TF-IDF + 3-Gram Vector", "desc": "Cosine Similarity + Domain Synonym Expansion", "results": r_vec, "latency_ms": t_vec},
        {"name": "4. Hybrid RRF Fusion", "desc": "Reciprocal Rank Fusion (Lexical + Vector)", "results": r_hyb, "latency_ms": t_hyb},
    ]

    fastest = min(methods, key=lambda x: x["latency_ms"])
    for m in methods:
        m["is_fastest"] = (m["name"] == fastest["name"])

    best_hit = r_hyb[0] if r_hyb else (r_vec[0] if r_vec else (r_fts[0] if r_fts else None))
    if best_hit:
        reply = f"[{best_hit['code']} | {best_hit['cat']}] {best_hit['content']}"
    else:
        reply = "No matching incident log found across any retrieval index."

    return jsonify({
        "methods": methods,
        "fastest_method": fastest["name"],
        "fastest_ms": fastest["latency_ms"],
        "best_method": "4. Hybrid RRF Fusion",
        "chatbot_reply": reply
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
