SELECT COUNT(*) FROM parks
WHERE {{LLMMap('How many states?', 'parks::Location')}} > 1