"Use vundle to manage plugin, required turn file type off and nocompatible
filetype off
set nocompatible
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
"Let vundle manage vundle, required
Plugin 'gmarik/Vundle.vim'
Plugin 'altercation/vim-colors-solarized'
Plugin 'rakr/vim-one'
Plugin 'tyrannicaltoucan/vim-quantum'
Plugin 'blueshirts/darcula'
"Plugin 'nightsense/cosmic_latte'
Plugin 'soft-aesthetic/soft-era-vim'
Plugin 'loremipsum'
"Plugin 'tomasr/molokai'
"Plugin 'itchyny/lightline'
"Plugin 'dunstontc/vim-vscode-theme'
"Plugin 'blueshirts/darcula'


call vundle#end()
filetype plugin indent on

"Show line number, command, status line and so on
set history=1000
set ruler
set number
set showcmd
set showmode
set laststatus=2
set cmdheight=2
set scrolloff=3

"Fill space between windows
set fillchars=stl:\ ,stlnc:\ ,vert:\ 

"Turn off annoying error sound
set noerrorbells
set novisualbell
set t_vb= 

"Turn off splash screen
set shortmess=atI

"syntax and theme
syntax on
colorscheme delek
"colorscheme one
"colorscheme quantum
"colorscheme darcula
"colorscheme cosmic_latte 
"colorscheme soft-era 
set background=dark
set cursorline
"hi CursorLine cterm=NONE ctermbg=lightgreen ctermfg=white
hi CursorLine gui=underline cterm=underline
set cursorcolumn
hi Cursorcolumn cterm=NONE ctermbg=240 ctermfg=NONE guibg=NONE guifg=NONE

"Configure backspace to be able to across two lines
set backspace=2
set whichwrap+=<,>,h,l

"Tab and indent
set expandtab
set smarttab
set shiftwidth=4
set tabstop=4
set autoindent
set cindent

"Files, backups and encoding
set nobackup
set noswapfile
set autoread
set autowrite
set autochdir
set fileencodings=ucs-bom,utf-8,utf-16,gbk,big5,gb18030,latin1
set fileformats=unix,dos,mac
set encoding=utf-8
filetype plugin on
filetype indent on

"Text search and replace
set showmatch
set matchtime=2
set hlsearch
set incsearch
set ignorecase
set smartcase
set magic
set lazyredraw
set nowrapscan
set iskeyword+=_,$,@,%,#,-,.

"Gvim config
if has("gui_running")
	colorscheme desert
endif
"set guifont=DejaVu\ Sans\ Mono:h15:cANSI:qDRAFT
set guifont=Consolas:h15:cANSI:qDRAFT
set guioptions=aegic
if has("autocmd")  
	au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif  
endif 
