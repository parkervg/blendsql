SELECT COUNT(*) FROM parks
WHERE {{LLMMap('How many states?', Location)}} > 1