function! myspacevim#before() abort
  let g:pydocstring_doq_path='~/.local/bin/doq'
  map <F4> :CondaChangeEnv<CR>
  autocmd BufNewFile,BufRead *.wl set syntax=wl
  autocmd BufNewFile,BufRead *.wls set syntax=wl
  autocmd BufNewFile,BufRead *.m set syntax=wl
  nnoremap <leader>gd :Gvdiff<CR>
  nnoremap gdh :diffget //2<CR>
  nnoremap gdl :diffget //3<CR>
endfunction

function! myspacevim#after() abort
endfunction
