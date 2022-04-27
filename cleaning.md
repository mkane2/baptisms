## Cleaning process
1. Open OCR transcription file in OpenRefine
2. Split columns by separator if possible
3. Change column names to date, child, parents, sponsors
4. Split parents and sponsors using make_columns.json (parents 1 and parents 2, sponsors 1, etc)
5. Fill down year as necessary
6. Remove blank rows or rows with non-relevant textual information (eg, "page XX", notes on a baptism, etc)
7. Edit parents and sponsors for punctuation as necessary (including [],;'?)
8. Split columns using split_columns.json
9. Apply first_transform.json, middle_transform.json, and last_transform.json to all columns
10. Cluster individual columns.  Extract the transformations, save to a separate file, and apply to all other columns.
11. Add new transformations to relevant file when done (first_transform.json, middle_transform.json, or last_transform.json)
12. Join all columns using join_columns.json
13. Export project as csv
14. Merge **all** name columns in exported csv into a single column
15. Create a **new** OpenRefine project using only the list of names
16. Cluster list of names
17. Extract transformations and save to a new file
18. Edit "Column 1" to "parents 1" etc and apply transformations to all columns in full OpenRefine project
19. Export full OpenRefine project as csv, save to transcriptions file, and push to github
