# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

force_color_prompt=yes
if [ -x /usr/bin/dircolors ]; then
	test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
	alias ls="ls --color=auto"
	alias dir="dir --color=auto"
	alias grep="grep --color=auto"
	alias fgrep="fgrep --color=auto"
	alias egrep="egrep --color=auto"
fi

alias python="python3.6"
alias ll="ls -al"
# alias for g++ support C++ standard 11, 14, 17, 20
alias g++11='g++ -std=c++11'
alias g++14='g++ -std=c++14'
alias g++17='g++ -std=c++17'
alias g++20='g++ -std=c++2a'

# Initialize Codex CLI
if [ -f "$HOME/.codexclirc" ]; then
    . "$HOME/.codexclirc"
fi
