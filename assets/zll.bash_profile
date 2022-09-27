# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs
alias cman='man -M /usr/share/man/zh_CN'
alias cheat='fc(){ cheat $1 | more;};fc'
