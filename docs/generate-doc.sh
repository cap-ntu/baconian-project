#!/usr/bin/env bash
source activate baconian || source activate py3.5 || source activate baconian-internal
# todo why here recommonmark package only work when baconian is the first one?
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR
cd $DIR
#cd  cwd/docs"${DIR}/../../baconian-doc"
DOC_DIR="$(dirname "$DIR")"
DOC_DIR="$(dirname "$DOC_DIR")"
echo $DOC_DIR
DOC_DIR="$DOC_DIR/baconian-doc"
echo $DOC_DIR
#cd $DOC_DIR
#echo $(pwd)
#NEW_DOC_DIR="$(readlink ${DOC_DIR} -f)"
#cd $DOC_DIR
#echo $NEW_DOC_DIR
sphinx-build -b html . $DOC_DIR
#cd $DIR/../../baconian-doc
#cd /home/dls/CAP/baconian-doc
cd $DOC_DIR
git add .
git commit -m"update"
git push -u origin master -f
ssh cap@155.69.144.246 "cd /home/cap/Websites/baconian-doc && git pull origin master -f"
