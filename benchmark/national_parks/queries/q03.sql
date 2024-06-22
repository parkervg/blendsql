SELECT "Location", "Name" AS "Park Protecting Ash Flow" FROM parks
WHERE "Name" = {{
  LLMQA(
    'Which park protects an ash flow?',
    context=(SELECT "Name", "Description" FROM parks),
    options="parks::Name"
  )
}}