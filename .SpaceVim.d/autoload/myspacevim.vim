function! myspacevim#before() abort
  let g:pydocstring_doq_path='~/.local/bin/doq'
  map <F4> :CondaChangeEnv<CR>
endfunction

function! myspacevim#after() abort
endfunction
