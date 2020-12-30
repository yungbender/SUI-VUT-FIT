#/usr/bin/env bash

submission=$1
workdir=$2

team_names="xbartl06 xberan43 xbulic02 xcalad01 xdacik00 xdemia00 xdizov00 xdufko02 xerlic00 xersek00 xfolen00 xforto00 xfrejl00 xhanak33 xhanze10 xhavli46 xhrivn02 xhrusk25 xhudec30 xhurta03 xchalo16 xirzha00 xjance00 xjavor18 xkabac00 xkapis00 xkolar71 xkotra02 xkrajc17 xkruty00 xkubov06 xkucer95 xkukuc04 xmatea00 xmrazi00 xnecpa00 xnekut00 xolsak00 xpalac03 xpavlu10 xpeska05 xpetru15 xpiste05 xpiwow00 xpomkl00 xreich06 xrysav27 xsadil06 xsalgo00 xsaman02 xsedla0v xsedla1d xskali16 xsobol04 xspane04 xsvacd00 xtrefi02 xtrnen03 xtrste00 xtulus00 xvagal00 xvostr08 xwilla00"

die () {
    echo "$*" >&2
    exit 1
}

[ $# -eq 2 ] || die "Usage: SUBMISSION WORKDIR"
[ $workdir ] || die "Workdir has to be of nonzero length"


[ -f "$submission" ] || die "Submission \"$submission\" not found"

fn=$(basename -- "$submission")
ext=${fn##*.}
login=${fn%.*}

[ $ext = "zip" ] || die "Unexpected file extension \"$ext\"" 
if echo "$team_names" | grep -w "$login" > /dev/null
then
    :
else 
    die "Unknown login \"$login\"" 
fi


mkdir -p $workdir

unzip -q "$submission" -d "$workdir" || exit $?

doc_location="$workdir/$login.pdf" 
[ -f "$doc_location" ] || die "Missing documentation \"$doc_location\""

if [ -f $workdir/$login.py ] ; then
    impl=$workdir/$login.py
    grep "except\s*:" $impl && die "Found bare \`except\`"
    grep "except\s*Exception\s*:" $impl && die "Found \`except Exception\`"
    grep "except\s*BaseException\s*:" $impl && die "Found \`except BaseException\`"
elif [ -d $workdir/$login ] ; then
    impl=$workdir/$login
    grep -r "except\s*:" $impl && die "Found bare \`except\`"
    grep -r "except\s*Exception\s*:" $impl && die "Found \`except Exception\`"
    grep -r "except\s*BaseException\s*:" $impl && die "Found \`except BaseException\`"
else
    die "Missing implementation"
fi 


nb_games=10
echo "### Running $nb_games games" 

git clone -q https://github.com/ibenes/dicewars.git $workdir/repo
cp -r $impl $workdir/repo/dicewars/ai/
cd $workdir/repo
mkdir ../logs
export PYTHONPATH=$PWD:$PYTHONPATH

python3 ./scripts/dicewars-ai-only.py -r -b 11 -o 22 -s 33 -c 44 -n $nb_games -l ../logs --ai $login xlogin42 dt.ste nop
