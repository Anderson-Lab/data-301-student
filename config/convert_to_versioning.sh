SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for file in `find . -name "*.ipynb" -print`; do
echo $file
filename="${file%.*}"
jupytext --to markdown --output "$filename.md" "$filename.ipynb"
python - <<EOF
import json
contents = json.loads(open("$filename.ipynb").read())
contents["metadata"]["jupytext"] = {"formats": "ipynb,py,md"}
new_content = json.dumps(contents, indent=4, sort_keys=True)
open("$filename.ipynb","w").write(new_content)
EOF
jupytext --to py --output $filename.py $filename.ipynb
jupytext --test -x $filename.ipynb --to py
done;
#jupytext --to markdown --output /results/simple-nb.md /jupyter-git/simple-nb.ipynb

IFS=$SAVEIFS