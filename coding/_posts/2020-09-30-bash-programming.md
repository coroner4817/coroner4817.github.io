---
layout: post
title: Bash Code Snippets
date: 2020-09-30 05:26 -0400
description: >
  Some code snippets for bash programming
image:
  path: "/assets/img/blog/bash.jpg"
related_posts: []
---

Bash scripts doing automation task can boost the productivity of daily work. I found myself really enjoy writing bash script. So here is some code snippets I learned. 

### head -1
```shell
#!/bin/bash
```

### Output redirection
```shell
xdg-open >/dev/null
```

### grep with sort
```shell
function grepsort {
  grep -rhi $1 | sort
}
```

### Multiple regex grep
```shell
cat file | egrep -o --color=never --line-buffered [PATTERN] | egrep -o --color=never --line-buffered [PATTERN]
```

### Redirect multiple options
```shell
function grepMulti{
  grep -i -e $@
}
```

### if condition
```shell
test -n "$1" # test if not empty
test -z "$1" # test if empty
[ "$1" == "-t" ] # test if equal
[ "$1" != "-t" ] # test if not equal
[ ${VAR: -1} == "1" ] # test if last char is 1
(($VAR1 > $VAR2)) # compare number
[ -d "$PATH" ] # test if path exist
[ $VAR -gt 0 ] # test if greater than
[[ "1234a" == "123"* ]] # test if regex matched
```

### Obtain the program that is in focus
```shell
W=`xdotool getactivewindow`
N=`xprop -id ${W} |awk '/WM_CLASS/{print $4}'`
```

### Keep the first line 
```shell
find . -name "*cpp*" | sed -n '1p'
```

### cut command
```shell
echo "1:2:3" | cut -d: -f2 # select the second 
echo "1:2:3" | cut -d: -f2- # select the part after first
echo "1:2:3" | cut -d':' -f2,1 # select the first and second  
echo "1:2:3" | cut -c -4 # select the part before 5th char
```

### String append
```shell
VAR1=$VAR'1'
```

### while waiting for something to be ready 
```shell
while [ -n "$cmd" ]; do
done
```

### if we have error
```shell
if [ $? -ne 0 ]; then
  exit 1
fi
```

### Disable Ctrl+c 
```shell
trap '' 2
# do your funky business
trap 2
```

### sed command
```shell
sed -i 's/[PATTERN]/'"$VAR"'/' file # replacement or removement in a file
```

### Iterate a list of grep result
```shell
RESULT=$(find . -name "*cpp*")
CNT=$(echo "$RESULT" | wc -l)
if test -n "$RESULT"; then
  ARR_RESULT=($RESULT)
  for i in "${ARR_RESULT[@]}"
  do
    echo $i
  done

  for (( i=0; i<$CNT; i++ ))
  do
    echo ${RESULT[$i]}
  done

fi
```

### Get runtime path
```shell
BASH_SOURCE_PATH=$(dirname "$BASH_SOURCE")
EXEC_PATH=$(pwd)
```

### Construct Array
```shell
ARR=()
ARR+=("1")
ARR+=("2")
```

### grep with color
```shell
# -e allow endl char
# -E allow extended regex patten so you don't need to escape some special character
# |$ force output of all lines, even not matched
VAR=("hello")
echo -e "hello\nworld" | GREP_COLOR='1;35' grep -E ''"$VAR"'|$' --color=always
```

### Share variable from different shell process
```shell
declare -a array
function fillArray {
  # name ref a variable
  declare -n arrayName=$1
  for i in {0..5}
  do
    arayName+=("item $i")
  done
}
fillArray array
echo "item count: ${#array[@]}"

# or we can write the shared variable to a tmp file
```

### Write function to handle piped input
```shell
function color_grep {
  IFS=''
  cat |
  while read line; do
    if [[ $line == "hello"* ]]; then
      echo $line | GREP_COLOR='1;33' grep -E '.*|$' --color=always
    fi
  done
}
```

### Trim endl
```shell
tr -d " \t\n\r"
```

### Flat a multi-line variable
```shell
flat=$(echo $MULTI)
```

### Run python in bash
```shell
VAR=("world")
PY_SCRIPT="temp.py"
touch $PY_SCRIPT
echo "
print(\"Hello\", sys.argv[1])
" > $PY_SCRIPT
python $PY_SCRIPT "$VAR"
rm $PY_SCRIPT
```

### Avoid escape
```shell
# wrapped text doesn't need to escape special char
# work with regex and sed
'I'\''m a s@fe $tring which ends'

# tricky way to generate string with escape characters
$ # This string 'has single' "and double" quotes and a $
$ !:q
https://til.simonwillison.net/til/til/bash_escaping-a-string.md
```

### Before you cd to somewhere else
```shell
pushd >/dev/null
popd >/dev/null
```

### C++ related
```shell
#demangle compiled symbol
nm --demangle $OBJ_FILE | egrep --line-buffered "$CLASS_NAME::"

# remove comment
https://stackoverflow.com/a/241506
```