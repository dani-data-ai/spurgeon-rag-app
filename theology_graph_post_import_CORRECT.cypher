// ========================================================================
// CORRECT Post-Import Script for YOUR Theology Graph
// 8M nodes, 181M edges, Node label only, RAG use case
// ========================================================================

// === STEP 1: VALIDATION (Run first to verify import) ===

// Count total nodes and edges
MATCH (n:Node) RETURN count(n) AS total_nodes;
// Expected: 7,955,943

MATCH ()-[r]->() RETURN count(r) AS total_edges;
// Expected: 180,853,750

// Check node distribution by source file
MATCH (n:Node)
RETURN n.source_file AS source, count(*) AS nodes
ORDER BY nodes DESC;

// Check relationship types
MATCH ()-[r]->()
RETURN r.relationship_type AS type, count(*) AS count
ORDER BY count DESC
LIMIT 20;

// Check for orphan nodes (nodes with no edges)
MATCH (n:Node)
WHERE NOT (n)--()
RETURN count(n) AS orphan_nodes;

// Verify cross-file edges exist
MATCH (a:Node)-[r]-(b:Node)
WHERE a.source_file <> b.source_file
RETURN a.source_file AS file1, b.source_file AS file2, count(*) AS cross_file_edges
ORDER BY cross_file_edges DESC
LIMIT 10;


// === STEP 2: CREATE INDEXES (CRITICAL FOR PERFORMANCE) ===
// Without these, queries will timeout on 8M nodes!

// Index on text for full-text search (RAG queries)
CREATE FULLTEXT INDEX node_text_fulltext IF NOT EXISTS
FOR (n:Node) ON EACH [n.text];

// Index on source_file for filtering
CREATE INDEX node_source_file IF NOT EXISTS
FOR (n:Node) ON (n.source_file);

// Index on entities for entity-based queries
CREATE INDEX node_entities IF NOT EXISTS
FOR (n:Node) ON (n.entities);

// Index on keywords for keyword search
CREATE INDEX node_keywords IF NOT EXISTS
FOR (n:Node) ON (n.keywords);

// Index on figures for figure-based queries
CREATE INDEX node_figures IF NOT EXISTS
FOR (n:Node) ON (n.figures);

// Index on parent_id for hierarchy queries
CREATE INDEX node_parent_id IF NOT EXISTS
FOR (n:Node) ON (n.parent_id);

// Index on relationship type for edge filtering
// Note: Relationship property indexes require Neo4j 5.0+
// If on older version, this will be skipped automatically
CREATE INDEX rel_type IF NOT EXISTS
FOR ()-[r]-() ON (r.relationship_type);

// Wait for indexes to come online
CALL db.awaitIndexes(300);


// === STEP 3: CREATE CONSTRAINTS ===

// No unique constraint on node IDs because your IDs are already globally unique
// (verified during CSV creation)

// But we can add existence constraints to ensure data quality:
// (Only if you want to enforce these - optional)

// CREATE CONSTRAINT node_has_source IF NOT EXISTS
// FOR (n:Node) REQUIRE n.source_file IS NOT NULL;


// === STEP 4: UPDATE GRAPH STATISTICS ===
// Helps Neo4j query planner optimize queries

CALL db.stats.collect();


// === STEP 5: TEST QUERIES FOR RAG USE CASE ===

// 1. Full-text search on theological concept
CALL db.index.fulltext.queryNodes('node_text_fulltext', 'justification faith')
YIELD node, score
RETURN node.text, node.source_file, score
ORDER BY score DESC
LIMIT 5;

// 2. Multi-hop traversal (sermon -> concept -> related sermon)
// Optimized for retrieval quality with weighted paths
MATCH path = (n1:Node)-[r*1..5]-(n2:Node)
WHERE n1.text CONTAINS 'justification'
  AND n2.text CONTAINS 'sanctification'
  AND n1.source_file CONTAINS 'spurgeon'
WITH path, n1, n2,
     reduce(totalWeight = 0.0, rel IN relationships(path) | totalWeight + rel.weight) AS pathWeight
RETURN n1.text AS start, n2.text AS end,
       pathWeight, length(path) AS hops
ORDER BY pathWeight DESC, hops ASC
LIMIT 10;

// 3. Cross-file reference check
MATCH (spurgeon:Node)-[r]-(reference:Node)
WHERE spurgeon.source_file CONTAINS 'spurgeon'
  AND reference.source_file CONTAINS 'reference'
RETURN spurgeon.text, type(r), reference.text
LIMIT 5;

// 4. Entity-based search
MATCH (n:Node)
WHERE n.entities CONTAINS 'Calvin'
RETURN n.text, n.source_file, n.entities
LIMIT 10;

// 5. Hierarchical query (parent-child relationships)
// Note: Your edges use relationship_type property, not typed relationships
MATCH (child:Node)-[r]->(parent:Node)
WHERE r.relationship_type = 'parent'
  AND parent.text CONTAINS 'sermon'
RETURN parent.text, collect(child.text)[0..3]
LIMIT 5;


// === STEP 6: MEMORY & PARALLEL PROCESSING TUNING ===
// (Add these to neo4j.conf, not Cypher)

// MEMORY ALLOCATION (28GB total RAM):
// dbms.memory.heap.initial_size=8G
// dbms.memory.heap.max_size=8G
// dbms.memory.pagecache.size=14G          # Increased for 181M edges
// (Leaves ~6GB for OS + other processes)

// PARALLEL PROCESSING (20 CPU cores, 7 workers max):
// dbms.threads.worker_count=7             # Your max worker limit
// dbms.query.parallel.enabled=true        # Enable parallel query execution
// dbms.cypher.parallel.runtime.enabled=true

// MULTI-HOP QUERY OPTIMIZATION (RAG priority):
// dbms.transaction.timeout=180s           # Allow long multi-hop queries
// dbms.memory.transaction.global_max_size=2G  # Per-query memory limit
// cypher.forbid_exhaustive_shortestpath=false # Allow deep traversals
// cypher.min_replan_interval=10s          # Adaptive query planning


// === STEP 7: BACKUP (Run from shell, not Cypher) ===
// neo4j-admin database dump theology_graph --to-path=/backups/


// ========================================================================
// DONE! Your graph is ready for RAG queries
// ========================================================================
