import embedding_analysis

if __name__ == '__main__':
    
    emb = 'LPP'
    '''
    embedding_analysis.main(channels_next=2, emb=emb)
    embedding_analysis.main(channels_next=4, emb=emb)
    embedding_analysis.main(channels_next=8, emb=emb)
    embedding_analysis.main(channels_next=16, emb=emb)
    '''
    embedding_analysis.main(channels_next=20, emb='LE')
    embedding_analysis.main(channels_next=20, emb='PCA')
    embedding_analysis.main(channels_next=20, emb='LPP')
    embedding_analysis.main(channels_next=20, emb='GPLVM')
    embedding_analysis.main(channels_next=20, emb='TSNE')
