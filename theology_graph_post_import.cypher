
// ===============================
// Theology Graph Post-Import Script
// ===============================

// === 1. Validation Queries ===

// Count nodes by label
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC;

// Count relationships by type
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count ORDER BY count DESC;

// Sample connectivity check
MATCH (p:Paragraph)-[:HAS_CHILD]->(s:Sentence) RETURN p.text, s.text LIMIT 5;

// Index check
CALL db.indexes();


// === 2. Alias Resolution ===

// Find potential aliases
MATCH (a:Author)
WHERE a.name CONTAINS "Owen"
RETURN a.name, count(*) ORDER BY count DESC;

// Merge aliases manually (requires APOC)
MATCH (a1:Author {name: "J. Owen"}), (a2:Author {name: "John Owen"})
CALL apoc.refactor.mergeNodes([a1, a2], {properties: "combine"})
YIELD node
RETURN node;

// Propagate alias to connected nodes
MATCH (a:Author)-[r:WRITTEN_BY]->(d:Document)
WHERE a.name IN ["J. Owen", "John Owen"]
SET a.name = "John Owen"
RETURN a.name, count(r);


// === 3. Semantic Cleanup ===

// Normalize dates
MATCH (d:Document)
WHERE d.date STARTS WITH "175"
SET d.date = date(d.date)
RETURN d.title, d.date LIMIT 5;

// Tag themes by keyword
MATCH (s:Sentence)
WHERE s.text CONTAINS "mercy"
MERGE (t:Theme {name: "Divine Mercy"})
MERGE (s)-[:HAS_THEME]->(t);

// Deduplicate sentences (requires APOC)
MATCH (s:Sentence)
WITH s.text AS txt, collect(s) AS nodes
WHERE size(nodes) > 1
CALL apoc.refactor.mergeNodes(nodes, {properties: "discard"})
YIELD node
RETURN node.text;


// === 4. Production Hardening ===

// Create indexes
CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name);
CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title);
CREATE INDEX sentence_text IF NOT EXISTS FOR (s:Sentence) ON (s.text);

// Add constraints
CREATE CONSTRAINT unique_author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT unique_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Backup reminder (external command)
// neo4j-admin dump --database=theology_graph --to=/backups/theology_graph.dump
