"""Original result-table formatter; manuscript rewrite functions are omitted."""
def table(label,caption,headers,rows):
    esc=lambda s:str(s).replace('_',r'\_')
    return '\n'.join([r'\begin{table}[!htbp]\centering\scriptsize',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+'l'*len(headers)+r'}\toprule',' & '.join(headers)+r'\\\midrule',*[' & '.join(esc(x) for x in row)+r'\\' for row in rows],r'\bottomrule\end{tabular}\end{table}'])
