paper.pdf: paper.md paper.bib
	pandoc --filter pandoc-citeproc --bibliography=paper.bib -s paper.md -o paper.pdf
