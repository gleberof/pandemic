def feat_imp(cols, fi):
    '''
    Function for amazing showing of feature importances
    
    Input:
    1) cols - list of feature names
    2) fi - np.array of feature importances
    
    Output:
    1) Table with features and their importances;
    2) Vizualization over barplot.
    '''
    
    import numpy as np
    from statsmodels.iolib.table import SimpleTable
    
    fi = np.round(fi, 3)
    indices = np.argsort(fi)[::-1]
    cols = [cols[i] for i in indices]
    
    
    print(SimpleTable(np.append([cols], [fi], axis=0).T,
                      ['Feature','Importance']))
    
    all_colors = list(plt.cm.colors.cnames.keys())
    c = np.random.choice(all_colors, fi.shape[0], replace=False)
    
    plt.figure()
    plt.title('Feature importances')
    plt.bar(range(fi.shape[0]), fi[indices], color=c, width=.5)
    plt.xticks(range(fi.shape[0]), cols, rotation=45)
    plt.show();
