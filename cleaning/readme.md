# OpenRefine/Regex cheatsheet

| type | GREL | explanation |
| --- | --- | --- |
| facet | value.contains(PATTERN) | true/false facet for a particular pattern (page, witnesses, wife) |
| transform | value.replace(PATTERN, "") | replace pattern text with nothing |
| regex | /^PATTERN/ | only match the pattern if it appears at the START of a string |
| regex | /PATTERN$/ | only match the pattern if it appears at the END of a string |
