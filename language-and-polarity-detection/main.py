import language_train_model as language
import sentiment
import ISO639_2 as codes
import sys
import argparse

__LANG_PATH__ = 'data/languages/paragraphs'
__SENT_PATH__ = 'data/movie_reviews/txt_sentoken'

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--ldata', help='Language data', default=__LANG_PATH__)
   parser.add_argument('--sdata', help='Sentiment data', default=__SENT_PATH__)
   args = vars(parser.parse_args())

   sentence = input("Sentence: ")
   language_results = language.get_language([sentence], args['ldata'])
   for result in language_results:
      print('The language of the sentence is '+codes.ISO639_2[result]+'.')
      if(result == 'en'):
         for result in sentiment.sentiment([sentence], args['sdata']):
            print('The sentiment of the sentence is '+('positive.' if result=='pos' else 'negative.'))
