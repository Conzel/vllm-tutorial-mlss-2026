MAIN := main

.PHONY: all watch clean distclean

all:
	latexmk -r latexmkrc $(MAIN).tex

watch:
	latexmk -r latexmkrc -pvc $(MAIN).tex

clean:
	latexmk -r latexmkrc -c $(MAIN).tex

distclean:
	latexmk -r latexmkrc -C $(MAIN).tex
	rm -rf build _minted*
