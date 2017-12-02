import gensim

if __name__ == '__main__':
    word2vec_path = "we_wiki_300_5"
    model = gensim.models.KeyedVectors.load(word2vec_path)
    model.init_sims(replace = True)


    model.save('normed_we_wiki_300_5')
