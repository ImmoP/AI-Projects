# Database Indexing Basics

A database index is a data structure, usually a B-tree, that lets the query
planner find rows without scanning the entire table. Without an index on the
primary key or a frequently filtered column, every SQL query that filters or
joins on that column forces a full table scan, which grows slower as the
table grows.

The query planner decides whether to use an index based on cardinality: how
many distinct values a column has relative to the number of rows. A column
with low cardinality, like a boolean flag, rarely benefits from an index,
because the planner would still need to read most of the table anyway.

Indexes are not free. Every insert or update has to maintain the B-tree
structure, which adds disk I/O and slows down writes. This is the classic
tradeoff in database indexing: faster reads at the price of slower writes and
extra storage for the index itself.
