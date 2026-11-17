import sqlite3
import random

CORE_RECORDS = [
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

SUBSYSTEMS = [
    ("NET-BGP", "Network Routing", "BGP peer session flapping on autonomous system border router"),
    ("K8S-POD", "Kubernetes Cluster", "CrashLoopBackOff on ingress controller pod during rolling deployment"),
    ("PG-VACUUM", "PostgreSQL Storage", "Transaction ID wraparound warning on high-throughput telemetry table"),
    ("JVM-GC", "Java Runtime", "Stop-the-world G1GC pause exceeding 800ms SLA threshold under heap pressure"),
    ("TLS-HANDSHAKE", "Cryptography", "TLS 1.3 cipher suite negotiation failure between edge proxy and upstream"),
    ("KAFKA-LAG", "Event Streaming", "Consumer group offset lag increasing on partitioned telemetry topic"),
    ("USB-XHCI", "Kernel Drivers", "xHCI host controller ring expansion timeout on DMA scatter-gather transfer"),
    ("GRPC-DEADLINE", "Microservices", "gRPC status DEADLINE_EXCEEDED during bidirectional streaming RPC call"),
    ("FS-EXT4", "Linux Filesystem", "Journal commit I/O bottleneck on NVMe namespace under synchronous fsync load"),
    ("MQTT-BROKER", "IoT Telemetry", "Keep-alive ping timeout disconnecting edge sensor fleet over cellular link")
]

CAUSES = [
    "caused by ephemeral port exhaustion on NAT gateway",
    "triggered by unaligned memory access in native C shared library",
    "due to MTU mismatch dropping jumbo frames across VXLAN tunnel",
    "resulting from thread pool starvation in blocking I/O worker",
    "linked to L3 cache thrashing across NUMA socket boundary",
    "stemming from stale DNS A-record caching in local stub resolver",
    "induced by TCP SYN backlog overflow during traffic spike",
    "traced to dirty page writeback throttling in virtual memory manager"
]

FIXES = [
    "Mitigation: Tune sysctl net.ipv4.ip_local_port_range and enable tcp_tw_reuse.",
    "Mitigation: Recompile binary with -O2 alignment flags and inspect core dump with gdb.",
    "Mitigation: Clamp TCP MSS to 1360 bytes on virtual bridge interface.",
    "Mitigation: Offload blocking calls to bounded async worker queue with backpressure.",
    "Mitigation: Pin worker processes to dedicated CPU cores using taskset affinity mask.",
    "Mitigation: Lower TTL to 30s and configure health-checked upstream resolver.",
    "Mitigation: Increase somaxconn queue depth and enable SYN cookies in kernel config.",
    "Mitigation: Adjust vm.dirty_ratio and vm.dirty_background_ratio for lower latency."
]

def build_database(db_path="arena.db", target_rows=1200):
    random.seed(42)
    records = list(CORE_RECORDS)
    
    for i in range(len(CORE_RECORDS) + 1, target_rows + 1):
        prefix, category, base_issue = random.choice(SUBSYSTEMS)
        cause = random.choice(CAUSES)
        fix = random.choice(FIXES)
        code = f"{prefix}-{i:04d}"
        node_hex = f"0x{random.randint(4096, 65535):04x}"
        content = f"[{node_hex}] {base_issue} {cause}. {fix}"
        records.append((i, code, category, content))

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS kb_standard")
    c.execute("DROP TABLE IF EXISTS kb_fts")

    c.execute("CREATE TABLE kb_standard (id INT PRIMARY KEY, code TEXT, category TEXT, content TEXT)")
    c.execute("CREATE VIRTUAL TABLE kb_fts USING fts5(id UNINDEXED, code, category, content, tokenize='porter')")

    c.executemany("INSERT INTO kb_standard VALUES (?, ?, ?, ?)", records)
    c.executemany("INSERT INTO kb_fts VALUES (?, ?, ?, ?)", records)
    conn.commit()
    conn.close()
    return records
