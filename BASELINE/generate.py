from utility.cross_validation import split_5_folds
from configx.configx import ConfigX


if __name__ == "__main__":
    
    #Douban Book ratings
    configx = ConfigX()
    configx.k_fold_num = 5 
    configx.rating_path = "../data/douban_book/user_item.dat"
    configx.rating_cv_path = "../data/douban_book_ratings_cv/"
    configx.dataset_name = "douban_book"
    configx.sep = '\t'
    split_5_folds(configx)


    
    #Douban Movie ratings
    configx = ConfigX()
    configx.k_fold_num = 5 
    configx.rating_path = "../data/douban_movie/user_item.dat"
    configx.rating_cv_path = "../data/douban_movie_ratings_cv/"
    configx.dataset_name = "douban_movie"
    configx.sep = '\t'
    split_5_folds(configx)

    #Yelp ratings
    configx = ConfigX()
    configx.k_fold_num = 5 
    configx.rating_path = "../data/yelp/user_item.dat"
    configx.rating_cv_path = "../data/yelp_ratings_cv/"
    configx.dataset_name = "yelp"
    configx.sep = '\t'
    split_5_folds(configx)
