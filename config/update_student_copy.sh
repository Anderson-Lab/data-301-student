cp -Rp * ../data-301-student/

# TODO go through every md file and remove certain sections
# BEGIN SOLUTION
# END SOLUTION
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for file in `find ../data-301-student/ -name "*.md" -print`; do
echo $file
filename="${file%.*}"
python - <<EOF
contents = open("$filename.md").read()
lines = []
started_solution = False
for line in contents.split("\n"):
  if "BEGIN SOLUTION" in line:
    started_solution = True
  elif "END SOLUTION" in line:
    started_solution = False
  elif started_solution == False:
    lines.append(line)
open("$filename.md","w").write("\n".join(lines))
EOF
done;