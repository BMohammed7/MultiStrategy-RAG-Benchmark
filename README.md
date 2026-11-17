# ⚡ DB-Retrieval-Arena: Multi-Strategy RAG & Index Benchmarker

A containerized full-stack benchmarking platform built to evaluate execution latency (in milliseconds) and retrieval relevance across **four distinct database and search architectures** side by side on a 1,200-record engineering incident corpus.

Designed to demonstrate the architectural trade-offs between sequential relational scans, inverted full-text indexes, TF-IDF vector cosine similarity, and hybrid Reciprocal Rank Fusion (RRF) inside a modular Python stack served by Gunicorn.

![DB-Retrieval-Arena dashboard comparing four retrieval methods on a typo-filled query](dashboard-screenshot.png)

---

## 📸 How to Read the Dashboard

The screenshot above shows one search run through all four methods at once. Here is what each part means:

* **Top right badge (`Corpus: 1200 Incident Logs`):** How many records are being searched.
* **Preset Stress Tests:** Four buttons that fill the search box with a ready-made example and run it in one click.
* **Search box + Run Benchmark:** Type your own question or error code and press Enter or click the button.
* **Green answer box:** The log the app would show a user as "the answer" (taken from the Hybrid RRF method). It also tells you which method was the fastest and how many milliseconds it took.
* **The four cards:** Each card is one search method looking at the *same* question. Every card shows:
  * the method name and a one-line description,
  * a **latency badge** (e.g. `0.605 ms`), which glows amber on the fastest method,
  * the best record that method found (its code and category),
  * a **Relevance** bar showing how confident the method is,
  * the text of the incident log it picked.
  * If a method finds nothing, the card says **0 Matches Retrieved** in red.

**What this particular screenshot shows:** the query `PyTrch CUDDA memoery crash` is full of typos. Only the **TF-IDF + 3-Gram Vector** card finds the right log (`AI-CUDA-88`, the PyTorch "CUDA out of memory" fix), because it compares small pieces of words instead of whole words. The keyword methods land on a Kubernetes log instead. Hybrid RRF ties between the two results here and shows the Kubernetes one, which is a real limitation of rank fusion when its two inputs disagree. SQLite FTS5 is the fastest card (0.605 ms) while the full-table `LIKE` scan takes about 3.5 ms.

---

## 🏗️ Retrieval Architectures Evaluated

Every query is run through four separate retrieval pipelines implemented in `retrieval_engines.py`:

| Strategy | Underlying Mechanism | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **1. SQL Keyword `LIKE`** | Sequential full-table scan across `kb_standard` (1,200 rows) | Exact substring matching with no index to build | Checks every row, so latency grows with data size; no tolerance for synonyms or typos |
| **2. SQLite FTS5 (BM25)** | Inverted index with Porter stemming (`kb_fts`) | Very fast exact-term and technical error code lookups | Needs matching words; misses synonyms and heavy typos |
| **3. TF-IDF + 3-Gram Vector** | Sparse TF-IDF vectors with cosine similarity, domain synonym expansion and character 3-grams | Handles natural phrasing ("phone lag") and spelling typos | More computation per query |
| **4. Hybrid RRF Fusion** | Reciprocal Rank Fusion combining the FTS5 and Vector ranked lists | Strong across both exact IDs and conversational queries | Runs both pipelines first; can tie or pick wrongly when the two disagree |

---

## 🛠️ Tech Stack

* **Application & WSGI Server:** Python 3.11, Flask 3.0, Gunicorn (2-worker configuration)
* **Storage & Indexing Engine:** SQLite3, FTS5 virtual tables, custom TF-IDF and character n-gram cosine similarity engine
* **Frontend Interface:** Jinja2 HTML template (`templates/index.html`), custom CSS (`static/style.css`), Tailwind CSS
* **Containerization:** Docker, Docker Compose

---

## 🚀 Prerequisites

No local Python environment or database installation is required. You only need:

* [Docker Desktop](https://docs.docker.com/get-docker/) (Windows/Mac) or Docker Engine + Compose v2 (Linux)

---

## 📥 Installation & Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/MultiStrategy-RAG-Benchmark.git
cd MultiStrategy-RAG-Benchmark
```

*(Or click **Code → Download ZIP** on GitHub and extract the archive.)*

**2. Build and start the Gunicorn container**

```bash
docker compose up --build
```

*(Run in the background with `docker compose up --build -d`.)*

**3. Open the benchmark dashboard**

Navigate to:

```text
http://localhost:5001
```

---

## 🧪 Stress-Testing the Indexes

Use the preset buttons at the top of the dashboard to see how the same corpus behaves under different kinds of queries:

* **1. Exact Key (`ERR-404-CAN`):** All four methods find the right log. Compare latency: the FTS5 inverted index is typically faster than the `LIKE` full-table scan.
* **2. Semantic Query (`Why is my phone app lagging when scrolling pictures?`):** All methods surface the Kotlin `RecyclerView` log, but compare the relevance bars and the runner-up results. The keyword methods also rank unrelated logs close behind.
* **3. Natural Language Incident (`container runs out of ram and crashes`):** Vector and Hybrid RRF return the Docker OOM log, while `LIKE` returns unrelated records.
* **4. Severe Typos (`PyTrch CUDDA memoery crash`):** Only the 3-gram vector engine recovers the PyTorch `CUDA out of memory` log. The keyword methods return unrelated Kubernetes logs, and Hybrid RRF ties between the two.

---

## 🗂️ What's in the Dataset

The database holds **1,200 incident logs**: 12 hand-written core logs plus 1,188 deterministically generated ones (seeded, so every run is identical).

**Core logs (12):**

| Code | Category | Topic |
|---|---|---|
| `ERR-404-CAN` | Automotive CAN Bus | Frame drops on wheel speed sensors, termination resistor |
| `SYS-QNX-09` | Embedded RTOS | QNX IPC deadlock between threads |
| `CLOUD-DOCKER-02` | DevOps & Containers | Docker exit code 137 (OOM killed) |
| `DB-SQLITE-11` | Database Engine | "Database is locked" error, WAL mode |
| `NET-REDIS-05` | Distributed Systems | Redis cluster split-brain, quorum |
| `AI-CUDA-88` | Machine Learning | PyTorch CUDA out of memory |
| `AV-ARINC-429` | Avionics Systems | ARINC 429 bus parity error |
| `GRID-SCADA-14` | Power Distribution | SCADA telemetry timeout over DNP3 |
| `MOB-KOTLIN-07` | Android Development | RecyclerView stutter, Kotlin Coroutines |
| `MEM-DRAM-03` | Computer Architecture | DRAM row buffer misses, high read latency |
| `SEC-JWT-21` | Backend Security | Valid JWT rejected due to clock drift (`nbf`) |
| `BIO-IK-19` | Biomechanics Sim | Inverse kinematics failing, marker tracking lost |

**Generated logs (1,188), numbered like `K8S-POD-0061`:**

| Code prefix | Category | Symptom |
|---|---|---|
| `NET-BGP` | Network Routing | BGP peer session flapping |
| `K8S-POD` | Kubernetes Cluster | CrashLoopBackOff on ingress controller pod |
| `PG-VACUUM` | PostgreSQL Storage | Transaction ID wraparound warning |
| `JVM-GC` | Java Runtime | G1GC stop-the-world pause over SLA |
| `TLS-HANDSHAKE` | Cryptography | TLS 1.3 cipher suite negotiation failure |
| `KAFKA-LAG` | Event Streaming | Consumer group offset lag |
| `USB-XHCI` | Kernel Drivers | xHCI ring expansion timeout |
| `GRPC-DEADLINE` | Microservices | gRPC `DEADLINE_EXCEEDED` |
| `FS-EXT4` | Linux Filesystem | Journal commit I/O bottleneck |
| `MQTT-BROKER` | IoT Telemetry | Keep-alive ping timeout on sensor fleet |

Each generated log also gets a random cause and mitigation sentence (for example "ephemeral port exhaustion on NAT gateway"), so the same words appear across many different subsystems.

---

## 🔍 More Queries to Try

Besides the four preset buttons, type any of these into the search box.

### Exact codes and technical terms (keyword search is strongest here)

| Query | Expected top match |
|---|---|
| `SYS-QNX-09` | QNX deadlock |
| `ARINC 429 parity error` | Avionics parity error |
| `PRAGMA journal_mode=WAL` | SQLite locking fix |
| `JWT 401 bearer` | JWT clock drift |
| `SCADA DNP3 timeout` | Power grid telemetry |
| `quorum master nodes` | Redis split-brain |
| `butterworth filter` | Biomechanics IK |
| `row buffer precharge` | DRAM latency |
| `K8S-POD-0061` | Generated Kubernetes log #61 |

### Natural language and synonyms

| Query | Expected top match |
|---|---|
| `my car wheel speed data keeps dropping` | `ERR-404-CAN` |
| `airplane bus parity problem` | `AV-ARINC-429` |
| `power grid substation not responding` | `GRID-SCADA-14` |
| `API rejecting tokens that look valid` | `SEC-JWT-21` |
| `gpu ran out of memory during training` | `AI-CUDA-88` |
| `two threads freeze waiting on each other` | `SYS-QNX-09` |
| `database locked when many users write` | `DB-SQLITE-11` |
| `skeleton back movement tracking lost` | `BIO-IK-19` |
| `container runs out of ram and crashes` | `CLOUD-DOCKER-02` (Vector and Hybrid) |

### Where the vector engine wins and keyword search gets distracted

Short, vague queries share words with many of the 1,188 generated logs, so the keyword methods grab unrelated records. Look at the **Vector** card for the right answer (Hybrid may tie with a keyword result).

| Query | Vector finds | Keyword methods tend to return |
|---|---|---|
| `phone lag` | `MOB-KOTLIN-07` | Unrelated "lag" or "pod" logs |
| `plane crash` | `AV-ARINC-429` | Kubernetes crash logs |
| `vehicle sensor frames missing` | `ERR-404-CAN` | MQTT or DRAM logs |

### Typos (character 3-grams handle these)

| Query | Expected top match |
|---|---|
| `sqllite databse locekd` | `DB-SQLITE-11` (`LIKE` and FTS5 return 0 matches) |
| `PyTrch CUDDA memoery crash` | `AI-CUDA-88` (Vector card only) |
| `dockr contaner OOM killd` | `CLOUD-DOCKER-02` |
| `Kotlinn corutines scrol` | `MOB-KOTLIN-07` |
| `redis clustr splt brain` | `NET-REDIS-05` |
| `tranformer substaton telemetry` | `GRID-SCADA-14` |

### Searching the generated logs

Each subsystem has over 100 near-identical logs, so these queries return *different* record IDs depending on the method. Great for comparing how each ranking works.

| Query | Subsystem |
|---|---|
| `kubernetes CrashLoopBackOff` | `K8S-POD` |
| `BGP peer flapping` | `NET-BGP` |
| `kafka consumer lag` | `KAFKA-LAG` |
| `TLS handshake failure` | `TLS-HANDSHAKE` |
| `G1GC pause` | `JVM-GC` |
| `gRPC DEADLINE_EXCEEDED` | `GRPC-DEADLINE` |
| `postgres vacuum wraparound` | `PG-VACUUM` |
| `MQTT keep-alive disconnect` | `MQTT-BROKER` |
| `USB xHCI timeout` | `USB-XHCI` |
| `ephemeral port exhaustion` | Matches the cause text across several subsystems |
| `MTU mismatch VXLAN` | Matches the cause text across several subsystems |

> **Tip:** Compare the four cards for each query. Watch which methods return **0 matches**, which has the lowest latency, and how the relevance scores differ.

---

## 🛑 Stopping the Container

To shut down the Gunicorn workers and remove the container network:

```bash
docker compose down
```

---

## 📂 Project Structure

```text
MultiStrategy-RAG-Benchmark/
├── app.py                    # Flask HTTP controller and REST benchmark endpoint
├── retrieval_engines.py      # RetrievalEngine class (LIKE, BM25, Cosine, RRF)
├── seed_data.py              # Deterministic 1,200-row incident log generator
├── requirements.txt          # Pinned Python dependencies (Flask, Gunicorn)
├── Dockerfile                # Python 3.11-slim container running Gunicorn
├── compose.yaml              # Docker Compose service and port configuration
├── dashboard-screenshot.png  # Screenshot shown at the top of this README
├── templates/
│   └── index.html            # Benchmark UI and client-side renderer
└── static/
    └── style.css             # Custom UI styling
```
