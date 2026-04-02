all: paper.pdf

bib:
	bibtex paper

paper.pdf: paper.tex macros.tex
	pdflatex paper

release:
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper

clean:
	rm -f paper.pdf paper.aux paper.out paper.log paper.loc paper.bbl paper.blg paper.soc

fullclean: clean
	rm -f *~
