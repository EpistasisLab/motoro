-- Runs once, on first boot of an empty data volume.
--
-- The test suite calls init_schema(drop_first=True), so it must not share a
-- database with development data. Two databases, one instance.
CREATE DATABASE agentic_core_test;
