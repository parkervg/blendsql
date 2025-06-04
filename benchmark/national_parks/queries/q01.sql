SELECT "Name",
{{ImageCaption(Image)}} as "Image Description",
{{
    LLMMap(
        question='Size in km2?',
        context=Area
    )
}} as "Size in km" FROM parks
WHERE "Location" = 'Alaska'
ORDER BY "Size in km" DESC LIMIT 1