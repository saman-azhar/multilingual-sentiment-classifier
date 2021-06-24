#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing libraries
from pandas.core.frame import DataFrame
from textblob import TextBlob
import re
from nltk.stem.wordnet import WordNetLemmatizer
import preprocessor as p
import pickle
from importlib import resources
import io

#reading trained models
MODELS = ["tweets_model_MNB.pkl", "tweets_model_CNB.pkl", "tweets_model_LR.pkl", "tweets_model_SVC.pkl", "tweets_model_SGD.pkl"]

with resources.open_binary('multilingualsentimentclassifier', MODELS[0]) as fp:
    filename1 = fp.read()
loaded_model_MNB = pickle.loads(filename1)
with resources.open_binary('multilingualsentimentclassifier', MODELS[1]) as fp:
    filename2 = fp.read()
loaded_model_CNB = pickle.loads(filename2)
with resources.open_binary('multilingualsentimentclassifier', MODELS[2]) as fp:
    filename3 = fp.read()
loaded_model_LR = pickle.loads(filename3)
with resources.open_binary('multilingualsentimentclassifier', MODELS[3]) as fp:
    filename4 = fp.read() 
loaded_model_SVC = pickle.loads(filename4)
with resources.open_binary('multilingualsentimentclassifier', MODELS[4]) as fp:
    filename5 = fp.read() 
loaded_model_SGD = pickle.loads(filename5)

# list of stopwords of all 3 languages
eng_stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats", "whence", "whenever",
                 "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]

rom_urdu_stopwords = ['ai', 'ayi', 'hy', 'hai', 'main', 'ki', 'tha', 'koi', 'ko', 'sy', 'woh', 'bhi', 'aur', 'wo', 'yeh', 'rha', 'hota', 'ho', 'ga', 'ka', 'le', 'lye', 'kr', 'kar', 'lye', 'liye', 'hotay', 'waisay', 'gya', 'gaya', 'kch', 'ab', 'thy', 'thay', 'houn', 'hain', 'han', 'to', 'is', 'hi', 'jo', 'kya', 'thi', 'se', 'pe', 'phr', 'wala', 'waisay', 'us', 'na', 'ny', 'hun', 'rha', 'raha', 'ja', 'rahay', 'abi', 'uski', 'ne', 'haan', 'acha', 'nai', 'sent', 'photo', 'you', 'kafi', 'gai', 'rhy', 'kuch', 'jata', 'aye', 'ya', 'dono', 'hoa', 'aese', 'de', 'wohi', 'jati', 'jb', 'krta', 'lg', 'rahi', 'hui''karna', 'krna', 'gi', 'hova', 'yehi', 'jana', 'jye', 'chal', 'mil', 'tu', 'hum', 'par', 'hay', 'kis', 'sb', 'gy', 'dain', 'krny', 'tou', 'Mahnoor', 'Ali', 'Noor', 'MUHAMMAD', 'Mishael', 'MOHAMMAD', 'mariam', 'Tariq', 'Aisha', 'Sunny', 'faiza', 'waqas', 'Anam', 'Farooq', 'karen', 'Zahid', 'Ayesha', 'usman', 'rameen', 'Bilal', 'Unsa', 'hamza', 'Neha', 'Yasir', 'Rabia', 'adnan', 'Rida', 'Hammad', 'zainnah', 'Hassan', 'Ameria', 'saif', 'sarah', 'Saad', 'Asmi', 'Amir', 'Mizha', 'SAMEER', 'ruby', 'Babar', 'Areeba', 'JAVED', 'Zainab', 'Kashif', 'Momna', 'Ibrahim',
                      'Rue', 'idrees khan', 'Eraj', 'fazal', 'sara', 'Subhan', 'Krishma', 'moheem', 'Alayna', 'imran', 'USMAAN', 'Shehzad', 'kainat', 'Tahir', 'imama', 'irfan', 'Sana', 'umair', 'Ujalaa', 'naeem', 'nazy', 'KHAN', 'Niya', 'Hadier', 'Afifa', 'Shahid', 'murium', 'Asad', 'Zoha', 'Abdul', 'Lintah', 'aqasha', 'sonia', 'Rabeel', 'Zakia', 'James', 'Aanya', 'Bilal', 'Hanif', 'komal', 'SAR', 'hajra', 'farhan', 'dashti', 'Ifrah', 'Usama', 'Lintah', 'Xain', 'Leah', 'talat', 'yaseen', 'tooba', 'zee', 'Asma', 'Kabir', 'Hussain', 'Kheezran', 'chand', 'sana', 'riasat', 'Zian', 'Saima', 'Talha', 'mahrukh', 'physics', 'haniya', 'Faraz', 'mariyam', 'Beaconite', 'umara', 'jahanzaib', 'Zuny', 'Ali', 'Kazmi', 'sajida', 'Ejaz', 'Zia', 'moiz', 'ahmed', 'ALY', 'Owais', 'ATIF', 'Talal', 'sheryar', 'Ihtisham', 'Sufian', 'HASSAAN', 'IFTEE', 'mitho', 'Chaudhary', 'dad', 'Ghulam', 'Qadir', 'jamshed', 'saleem', 'sharif', 'Hansraj', 'rai', 'Shan', 'Aatif', 'Wishal', 'Maqbool', 'Ahmed', 'a haq', 'ansari', 'waseem', 'wakeel', 'khan', 'Jarrar', 'Faizan', 'daniyal', 'Wasay', 'Danial', 'noman', 'Mazhar', 'Ali', 'RAZA', 'Qais', 'Ranjhoo', 'Rauf', 'Shah', 'Aamir', 'Saleem', 'Fahid', 'Ash']

urdu_stopwords = ["ایم "," لگ رہا تھا "," بظاہر "," لگتا ہے "," دیکھا "," خود "," خود "," بھیجا "," سات "," متعدد "," گے "," شیڈ " , "شو", "دکھایا", "دکھایا گیا", "شو", "شو", "نمایاں", "نمایاں", "مماثل", "اسی طرح", "چونکہ", "چھ", "قدرے", " کوئی "," کسی طرح "," کسی "," کچھ "," کچھ "," کبھی "," کبھی "," کچھ "," کہیں "," جلد "," معذرت "," خاص ", "وضاحت", "پھر بھی", "رک", "مضبوطی سے", "ذیلی", "خاطر خواہ", "کامیابی", "کافی", "تجویز", "مدد", "یقینی"  , "لیا" , "لے" , "بتائیں" , "ٹینڈز" , "ویں" , "تھینکس" , "تھینکس" , "تھینکس" , "تھات" , "جس نے" , "وہیں" , " اس کے بعد "," اس کے ساتھ "," تھیریڈ "," لہذا "," اس "," وہاں "," اس "," وہاں "," وہاں "," وہاں "," وہاں " , "ہزار", "اس طرح", " ٹپ "," ایک ساتھ "," لیا "," کی طرف "," کی طرف "," آزمایا "," کوشش "," واقعی "," کوشش "," کوشش "," ای "," دو "," یو "," ان "," بدقسمتی سے "," جب تک "," اس کے برعکس "," غیر امکان "," سے "," پر "," اپ "," ہم "," استعمال " , "استعمال شدہ", "مفید", "مفید", "افادیت", "استعمال", "استعمال", "عام طور پر", "قدر", "مختلف"," کے ذریعے ", "مثلا" ," والیوم "," جلد "," بمقابلہ "," ڈبلیو "," مطلوب "," مطلوب "," ضائع "," راستہ "," شادی "," استقبال "," گئے "," نہیں تھے " ," جو بھی "," کیا کریں گے "," کیا "," کہاں سے "," جب بھی "," جہاں "," جبکہ "," جہاں "," جہاں "," پہیے "," جہاں "," چاہے "," وہم "," وہاں "," کس طرح "," جو بھی "," سارا "," کون کرے گا "," کون "," کون "," جس کا "," وسیع پیمانے پر "," تیار "," خواہش "," کے اندر "," بغیر "," آوارہ "," الفاظ "," دنیا "," نہیں "," ہاں "," ابھی "," یو ڈی ", "آپ" , "زیڈ" , "صفر" , "ایک" , "نہیں" , "اجازت" , "اجازت" , "الگ" , "نمودار" , "تعریف" , "مناسب" , "وابستہ" , "بہترین" , "بہتر" , "کیمون" , "سی" , "کینٹ" , "تبدیلیاں" , "واضح طور پر" , "کے بارے میں" , "نتیجہ" , "غور" , "غور" , "مطابقت پذیر" , "کورس" , "فی الحال" , "یقینی طور پر" , "بیان" , "باوجود" , "مکمل"," بالکل "," مثال "," جا رہے ہیں "," مبارکبادیں "," ہیلو "," مدد "," امید ہے "," نظر انداز "," انسمچ "," اشارہ "," اشارہ "," اشارہ " , "اندرونی" , "انسفر" , "یہ" , "رکھیں" , "رکھتا ہے" , "ناول" , "شاید" , "معقول" , "دوسرا" , "ثانوی" , "سمجھدار" , "سنجیدہ" , "سنجیدگی سے", "یقینی", "تیسرا", "مکمل", "اچھی طرح", "تین", "اچھی طرح سے", "حیرت", "کے بارے میں", "اوپر", " اوپر "," پار "," بعد "," بعد "," پھر "," کے خلاف "," سب "," تقریبا "," تنہا "," ساتھ "," پہلے ہی "," بھی "," اگرچہ " , "ہمیشہ" , "ہوں" , "آپس میں" , "درمیان" , "امونگینگ" , "رقم" , "ان" , "اور" , "دوسرا" , "کوئی" , "کسی بھی طرح" , "کسی" , " کچھ بھی "," بہرحال "," کہیں بھی "," ہیں "," آس پاس "," جیسے "," پیچھے "," بن "," بنے "," کیونکہ "," بن "," بنے " , "بن رہا", "رہا", "پہلے", "پہلے", "پیچھے", "ہونا", "نیچے", "ساتھ", "علاوہ", "کے درمیان", "پرے", "بل", " دونوں "," نیچے "," لیکن "," بہ "," کال "," کر سکتے ہیں "," نہیں کر سکتے "," کھچڑی "," کو "," کون "," کر سکے "," رونے " , "ڈی" , "وضاحت" , "تفصیل" , "کرو" , "ہو گیا "," نیچے "," واجب "," دوران "," ہر "," مثال کے طور پر "," آٹھ "," یا تو "," گیارہ "," اور "," کہیں اور "," خالی "," کافی ", "وغیرہ" , "بھی" , "کبھی" , "ہر" , "سب" , "سب کچھ" , "ہر جگہ" , "سوائے" , "چند" , "پندرہ" , "فیٹ" , "پُر" , "ڈھونڈیں "," آگ "," پہلے "," پانچ "," کے لئے "," سابقہ "," پہلے "," چالیس "," پایا "," چار "," سے "," سامنے "," بھرا ", "مزید" , "حاصل" , "دینا" , "جانا" , "تھا" , "ہے" , "ہنس" , "ہے" , "وہ" , "لہذا" , "اس" , "یہاں" , "اس کے بعد "," اس کے ذریعہ "," یہاں "," یہاں "," اس کا "," خود "," خود "," خود "," اس "," کیسے "," تاہم "," سو "," یعنی ", "اگر" , "ان" , "انک" , "در حقیقت" , "دلچسپی" , "میں" , "ہے" , "یہ" , "اس" , "خود" , "رکھیں" , "آخری" , "مؤخر الذکر" ," بعد میں "," کم سے کم "," کم "," لمیٹڈ "," بنا "," بہت سے "," ہوسکتا ہے "," میں "," اس دوران "," شاید "," چکی "," میرا ", "زیادہ" , "مزید" , "زیادہ تر" , "زیادہ تر" , "چال" , "زیادہ" , "لازمی طور پر" , "میرا" , "خود" , "نام" , "نام" , "نہ ہی" , "کبھی نہیں"," اس کے باوجود "," اگلا "," نو "," نہیں "," کوئی نہیں "," نون "," اور "," نہیں "," کچھ نہیں "," اب "," کہیں نہیں ", "آف" , "آف" , "اکثر" , "آن" , "ایک بار" , "ایک" , "صرف" , "پر" , "یا" , "دوسرے" , "دوسرے" , "بصورت دیگر" , "ہمارے" , "ہمارے" , "خود" ," آؤٹ "," اوور "," خود "," پارٹ "," فی "," شاید "," پلیز "," ڈال "," بلکہ "," دوبارہ "," اسی "," دیکھیں ", "لگتا ہے" , "لگتا ہے" , "بظاہر" , "لگتا ہے" , "سنجیدہ" , "متعدد" , "وہ" , "چاہئے" , "شو" , "سائیڈ" , "چونکہ" , "مخلص" , "چھ "," ساٹھ "," تو "," کچھ "," کسی طرح "," کسی "," کچھ "," کبھی "," کبھی "," کہیں "," پھر بھی "," ایسے "," نظام ", "لے" , "دس" , "سے" , "وہ" ," ان "," انہیں "," خود "," پھر "," وہاں "," وہاں "," اس کے بعد "," اس کے بعد "," لہذا "," اس میں "," اس کے بعد "," یہ "," وہ "," موٹوی "," پتلا "," تیسرا "," یہ "," وہ "," اگرچہ "," تین ", "کے ذریعے" , "بھر" , "تھرو" , "اس طرح" , "سے" , "ایک ساتھ" , "بھی" , "ٹاپ" , "طرف" , "کی طرف" , "بارہ" , "بیس" , "دو "," ان "," تحت "," جب تک "," اوپر "," پر "," ہم "," بہت "," کے ذریعے "," تھا "," ہم "," خیریت سے "," تھے ", "کیا" , "جو بھی" , "جب" , "کہاں سے" , "جب بھی" , "جہاں" , "جہاں" , "جب" , "جہاں" , "جہاں","ایک" , "کے بارے میں" , "اوپر" , "بعد" , "پھر" , "کے خلاف" , "آئین" , "سب" , "ام" , "ایک" , "اور" , "کوئی" , "نہیں ہیں" , "جیسے" , "ہو" , "کیونکہ" , "رہے" , "پہلے" , "ہونے" , "نیچے" , "کے درمیان" , " دونوں "," لیکن "," بہ "," کر سکتے ہیں "," قابل "," نہیں کر سکتے "," د "," کیا "," کیا "," نہیں "," نہیں "," کرتا ہے " , "نہیں" , "نہیں" , "کر رہا ہے" , "ڈان" , "نہیں" , "نیچے" , "دوران" , "ہر" , "کچھ" , "کے لئے" , "سے" , " مزید "," تھا "," ہینڈ "," نہیں "," تھا "," ہنس "," نہیں "," ہے "," ہیون "," نہیں "," وہ "," اس "," یہاں "," اس "," خود "," اسے "," خود "," اس "," کیسے "," میں "," اگر "," میں "," میں ", "میں" , "طاقتور" , "شاید" , "زیادہ" , "سب سے زیادہ" , "مستن" , "نہیں" , "میرا" , "خود" , "محتاج" , "ضرورت نہیں" , "نہیں" , "نہ" , "نہیں" , "اب" , "او" , "آف" , "آف" , "آن" , "ایک بار" , "صرف" , "یا" , "دوسرے" , "ہمارے" , " ہماری "," خود "," آؤٹ "," اوور "," اپنی "," دوبارہ "," ایس "," وہی "," شان "," شانت "," وہ "," وہ ", "چاہئے" , "چاہئے" , "نہیں" , "تو" , "کچھ" , "ایسے" , "ٹی" , "سے" , "وہ" , "وہ" ,"گے "," ان "," ان "," ان "," وہ "," خود "," تب "," وہاں ", "یہ" , "وہ" , "یہ" , "وہ" , "کے ذریعے" , "سے" , "بھی" , "تحت" , "جب تک" , "اپ" , "وی" , "بہت" , "تھے" ," تھا "," نہیں "," ہم "," تھے "," تھے "," نہیں تھے "," کیا "," جب "," کہاں "," کون "," جبکہ ", "کون" , "کسے" , "کیوں" , "کرے گا" , "ساتھ" , "جیتا" , "نہیں کرے گا" , "نہیں" , "نہیں" , "ی" , "آپ" , "آپ" , "آپ" , "آپ" , "آپ" , "آپ" , "آپ" , "خود" , "خود" , "ہو" , "وہ" , "وہ" "ایل ایل" , "وہ" , "یہ ہے" , "کیسا ہے" , "میں ہوں" , "میں ہوں گا" , "میں ہوں" , "میں ہوں" , "آئیے" , "چاہئے" , "وہ" , "وہ" , "وہ" , "وہاں" , "وہ" , "وہ" , "وہ" , "وہ" , "ہم" , "ہم ہوں گے" , "ہم" "ایل ایل" , "ہم" , "ہم" , "کیا" , "کب" , "کہاں ہیں" , "کون ہے", "کیوں" , "کیوں" , "قابل" , "مطابق" ," مطابق "," اس کے مطابق "," اس پار "," ایکٹ "," اصل میں "," شامل "," صفت "," متاثر "," متاثر "," اثر انداز "," بعد میں "," آہ ", "تقریبا" , "تنہا" , "ساتھ" , "پہلے ہی" , "بھی" , "اگرچہ" , "ہمیشہ" , "آپس میں" , "درمیان" , " "," دوسرا "," کسی کو بھی "," کسی بھی طرح "," اب "," کسی کو بھی "," کچھ بھی "," بہرحال "," ویسے بھی "," کہیں بھی "," بظاہر "," تقریبا ","کا اعلان کریں۔ ", "اٹھ", "ارد گرد", "ایک طرف", "پوچھنا", "پوچھ", "دستیاب", "دور", "خوفناک", "پیچھے", "بن گیا", " بن "," بنتا ہے "," بنتا "," پہلے "," شروع "," شروع "," شروعات "," شروع "," پیچھے "," یقین "," ساتھ "," علاوہ "," سے آگے " , "بائول" , "مختصر" , "مختصر طور پر" , "سی" , "سی اے" , "آیا" , "نہیں کر سکتے" , "نہیں" , "وجہ" , "وجوہات" , "کچھ" , "یقینی طور پر" , "کو" , "کام" , "آئے" , "آتا ہے" , "مشتمل" , "مشتمل" , "مشتمل" , "کانٹ" , "تاریخ" , "مختلف" , "کیا ہوا" , "نیچے کی طرف" ," ای "," ای ڈی "," ایدو "," اثر "," مثال کے طور پر "," اسی "," اسی "," یا تو "," کسی اور "," کہیں اور "," اختتام "," اختتامی " , "کافی" , "خاص طور پر" , "ات" , "وغیرہ" , "یہاں تک" , "کبھی" , "ہر" , "ہر ایک" , "سب" , "سب کچھ" , "ہر جگہ" , "سابق" , " سوائے "," ایف "," بعید "," ایف ایف "," پانچواں "," پہلے "," پانچ "," فکس "," فالوڈ "," فالونگ "," فالس "," سابق "," پہلے " , "آگے" , "ملا" , "فو آپ "," اس کے علاوہ "," جی "," دیا "," حاصل "," ملتا ہے "," حاصل "," دے "," دیا "," دیتا "," دے "," جاتا "," جاتا ہے " , "گیا", "ملا", "حاصل", "ہ", "ہوتا ہے", "مشکل سے", "ہیڈ", "لہذا", "یہاں", "یہاں", "یہاں", " اس کے بعد "," ہیک "," ہائے "," چھپا "," یہاں "," گھر "," بہرحال "," تاہم "," سو "," شناخت "," یعنی "," آئی ایم "," فوری " , "فورا" ," اہمیت "," اہم "," انک "," واقعی "," اشاریہ "," معلومات "," بجائے "," ایجاد "," اندرونی "," آئی ٹی ڈی "," یہ " , "جے" , "کے" , "رکھیں" , "رکھتا ہے" , "رکھا" , "کلو" , "کلومیٹر" , "جانتے" , "جانا جاتا" , "جانتا" , "ایل" , "بڑے پیمانے پر" , " آخری "," حال ہی میں "," بعد میں "," مؤخر الذکر "," بعد میں "," کم سے کم "," کم "," ایسا نہیں "," لات "," اجازت "," پسند "," پسند "," امکان " , "لائن", "چھوٹا"," دیکھو "," دیکھ "," لگ رہا ہے "," لمیٹڈ "," بنا "," بنیادی طور پر "," بنا "," بناتا ہے "," بہت " "شاید" , "ہوسکتا ہے" , "مطلب" , "مطلب" , "اس دوران" , "اس دوران" , "محض" , "ملیگرام" , "شاید" , "ملین" , "مس" , "ملی" , "مزید یہ "," زیادہ تر "," مسٹر "," مسز "," زیادہ "," مگ "," ضرور "," این "," نا "," نام "," یعنی "," نہیں "," این ڈی ", "قریب" , "قریب" , " لازمی طور پر "," ضروری "," ضرورت "," ضروریات "," نہ "," کبھی نہیں "," بہرحال "," نیا "," اگلا "," نو "," نوے "," کوئی نہیں "," غیر " , "کچھ نہیں" , "بہرحال" , "نون" , "عام طور پر" , "نمبر" , "مشہور" , "کچھ نہیں" , "کہیں نہیں" , "حاصل" , "حاصل" , "ظاہر" , "اکثر" , " اوہ "," اوکے "," اوکے "," پرانا "," چھوٹا ہوا "," ایک "," ایک "," پر "," آرڈر "," دوسرے "," بصورت دیگر "," باہر "," مجموعی طور پر " , "واجب" , "پی" , "صفحہ" , "صفحات" , "حصہ" , "خاص" , "خاص طور پر" , "ماضی" , "فی" , "شاید" , "رکھے ہوئے" , "براہ کرم" , " جمع "," ناقص "," ممکن "," ممکنہ طور پر "," ممکنہ طور پر "," پی پی "," بنیادی طور پر "," حال "," پہلے "," بنیادی طور پر "," شاید "," فوری طور پر "," فخر " , "فراہم کرتا ہے", "ڈال", "کیو", "کوئ", "جلدی", "کافی", "کیو", "ر", "میں "," جہاں "," جہاں "," چاہے "," کون "," جبکہ "," جہاں "," کون "," جو "," پوری "," کس "," کس "," کیوں " , "ساتھ", "کے اندر", "بغیر", "آپ", "آپ", "آپ", "خود", "خود" , "این" , "او" , "پی" , "ق" , "ر" , "ایس" , "ٹی" , "یو" , "وی" , "ڈبلیو" , "ایکس" , "ی" ," زیڈ "," شریک "," آپ ","ریسرچ آرٹیکل "," پیجکاؤنٹ "," سائٹ "," آئبید "," لیس "," لی "," او "," کوئ "," لاس "," اب "," اشتہار " , "عی" , "اے ایف" , "اگ" , "اج" , "ال" , "آن" , "او او" , "اپ" , "اے آر" , "اے وی" , "او" , "کلہاڑی"  , "بی این" , "بی پی" , "بی آر" , "بی ایس" , "بی ٹی" , "بی او" , "بی ایکس" , "سی " , "سی " , "سی" , "سی سی" , "سی ڈی" , " سی ای "," سییف "," سی جی "," چ "," سی آئی ", "سی زیڈ" , "ڈی " , "دا" , "ڈی سی" , "ڈی ڈی" , "ڈی" , "ڈی ایف" , "دی" , "ڈی جے" , "ڈی کے" , "ڈی ایل" , "ڈو" , " ڈی پی "," ڈرا "," ڈی ایس "," ڈی ٹی "," ڈو "," ڈی ایکس "," ڈائی "," ای  "," ای  "," ای اے "," ای سی "," ای ڈی "," ای ای " , "ای ایف" , "ای آئی" , "ایج" , "ایل" , "ایم" , "این" , "ای او" , "ایپی" , "ایق" , "ایر" , "ایس" , "ات" ," سابق "," فا ", "فو" ," ہائے "," ہو " , "آئک" , "یعنی" ," آئی جی "," آئی ایل "," ان "," آئی او "," آئی پی "," آئی کی "," آئی آر "," آئی وی "," آئیکس "," آئی آئی "," آئی ایس او "," جے جے " , "جونیئر" , "جے ایس" , "جے ٹی" , "جو" , "کی" , "کلو" , "کے جے" , "کلومیٹر" , "کو" , "ایل " , "لا" , "ایل بی" ," ایل ایف "," ایل جے "," ایل این "," لو "," ایل آر "," ایل ایس "," ایل ٹی "," ایم  "," ایم ایل "," ایم این "," مو "," ایم ایس " , "ایم ٹی" , "ایم یو" , "این " , "این سی" , "این ڈی" , "نی" , "این جی" , "نی" , "این جے" , "این ایل" , "این این" , "این آر" , " این ایس "," این ٹی "," نی "," او اے "," اوب "," او سی "," اوڈ "," آف "," اوگ "," او آئی "," اوج "," او ایل "," اوم " , "آن" , "او" , "اوق" , "یا" , "اوس" , "اوٹ" , "او" , "اوہ" , "بیل" , "اوز" , "پی " , "پی " , "پی " , "پی سی" , "پی ڈی" , "پی اے" , "پی ایف" ," پی ایچ "," پی آئی "," پی جے "," پی کے "," پی ایل "," پی ایم "," پی این "," پو "," پی کیو "," پی آر "," پی ایس "," پی ٹی ", "پو" , "پیی" , "کی جے جے" , "ق" , "آر " , "را" , "آر سی" , "آر ڈی" , "آر ایف" , "آر ایچ" , "ر" , "آر جے" , "آر ایل "," آر ایم "," آر این "," آر او "," آر کیو "," آر آر "," آر ایس "," آر ٹی "," رو "," آر وی "," آری "," ایس ٹو "," سا ", "ایس سی" , "ایس ڈی" , "ایس ای" , "ایس ایف" , "سیی" , "ایس جے" , "ایس ایل" , "ایس ایم" , "ایس این" , "ایس پی" , "اسکیور" , "ایس آر" , "ایس ایس" ," ایس ٹی "," سی ای "," ایس زیڈ "," ٹی  "," ٹی  "," ٹی  "," ٹی بی "," ٹی سی "," ٹی ڈی "," ٹی "," ٹی ایف "," ویں ", "ٹائی" , "ٹی جے" , "ٹی ایل" , "ٹی ایم" , "ٹی این" , "ٹی پی" , "ٹیکی" , "ٹی آر" , "ٹی ایس" , "ٹی ٹی" , "ٹی وی" , "ٹی ایکس" , "یو "," یوئی "," اوج "," یوکے "," ام "," ان "," یو او "," آپ "," یوٹ "," وا "," وا "," وی ڈی "," وائی ", "وی جے" , "وو" , "وو" , "وی کیو" , "وی ٹی" , "وو" , "ایکس " , "ایکس " , "ایکس " , "ایکس ایف" , "ایکس آئی" , "ایکس جے" , "ایکس کے" , "یٹ" , "زی" , "زیڈز","رن", "بلکہ", "آر ڈی", "آسانی سے", " واقعی "," حال ہی میں "," حال ہی میں "," ریف "," حوالہ جات "," متعلقہ "," قطع نظر "," احترام "," متعلقہ "," نسبتا, "," تحقیق "," بالترتیب "," نتیجہ " , "نتیجہ", "نتائج", "حق", "رن", "کہا", "دیکھا", "کہنا", "کہنا", "کہتا ہے", "سیکنڈ", "سیکشن", "دیکھیں"," دیکھیں"]

class MethodsForText(object):
    def text_processing_english(self, text):
        text = p.clean(text)

    #  Generating the list of words in the tweet (hastags and other punctuations removed)
        def form_sentence(text):
            text_blob = TextBlob(text)
            return ' '.join(text_blob.words)
        new_text = form_sentence(text)

        # Removing stopwords and words with unusual symbols
        def no_user_alpha(text):
            text_list = [ele for ele in text.split() if ele != 'user']
            clean_tokens = [t for t in text_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split(
            ) if word.lower() not in eng_stopwords]
            return clean_mess
        no_punc_text = no_user_alpha(new_text)

        # Normalizing the words in tweets

        def normalization(text_list):
            lem = WordNetLemmatizer()
            normalized_text = []
            for word in text_list:
                normalized_text = lem.lemmatize(word, 'v')
            return normalized_text

        return normalization(no_punc_text)

    def text_processing_roman_urdu(self, text):
        text = p.clean(text)

        # Generating the list of words in the tweet (hastags and other punctuations removed)
        def form_sentence(text):
            text_blob = TextBlob(text)
            return ' '.join(text_blob.words)
        new_text = form_sentence(text)

        # Removing stopwords and words with unusual symbols
        def no_user_alpha(text):
            text_list = [ele for ele in text.split() if ele != 'user']
            clean_tokens = [t for t in text_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split(
            ) if word.lower() not in rom_urdu_stopwords]
            return clean_mess
        no_punc_text = no_user_alpha(new_text)

        # Normalizing the words in tweets
        def normalization(text_list):
            lem = WordNetLemmatizer()
            normalized_text = []
            for word in text_list:
                normalized_text = lem.lemmatize(word, 'v')
            return normalized_text

        return normalization(no_punc_text)

    def text_processing_urdu(self, text):
        text = re.sub(r"\d+", " ", text)
        # English punctuations
        text = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]+""", " ", text)
        # Urdu punctuations
        text = re.sub(r"[:؛؟’‘٭ء،۔]+", " ", text)
        # Arabic numbers
        text = re.sub(r"[٠‎١‎٢‎٣‎٤‎٥‎٦‎٧‎٨‎٩]+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        # Remove English characters and numbers.
        text = re.sub(r"[a-zA-z0-9]+", " ", text)
        # remove multiple spaces.
        text = re.sub(r" +", " ", text)
        text = text.split(" ")
        # some stupid empty tokens should be removed.
        text = [t.strip() for t in text if t.strip()]

        return text

    def prediction(self, preprocessed_text):

        result_MNB = loaded_model_MNB.predict(preprocessed_text)
        result_LR = loaded_model_LR.predict(preprocessed_text)
        result_SVC = loaded_model_SVC.predict(preprocessed_text)
        result_SGD = loaded_model_SGD.predict(preprocessed_text)
        result_CNB = loaded_model_CNB.predict(preprocessed_text)

        sum = result_MNB + result_LR + result_SVC + result_SGD + result_CNB
        if(sum >= 3):
            return 'Negative'
        else:
            return 'Positive'

    def predict_sentiment(self, text: str, language: str)-> str:

        if(language == 'en'):
            text = self.text_processing_english(text)
            preprocessed_text = ''.join(text.lower())
        elif(language == 'ur'):
            preprocessed_text = self.text_processing_urdu(text)
            preprocessed_text = ''.join(text)
        elif(language == 'in'):
            text = self.text_processing_roman_urdu(text.lower())
            preprocessed_text = ''.join(text)
        else:
            print(
                'please choose one of these languages: "english: en", "urdu: ur" or "roman urdu: in"')
        return self.prediction([preprocessed_text])


class MethodsForDataframe(MethodsForText):

    def preprocess_tweet(self, row):
        text = row[0]
        return p.clean(str(text))

    def text_processing_english(self, tweet):
        #  Generating the list of words in the tweet (hastags and other punctuations removed)
        def form_sentence(tweet):
            tweet_blob = TextBlob(tweet)
            return ' '.join(tweet_blob.words)
        new_tweet = form_sentence(tweet)

        # Removing stopwords and words with unusual symbols
        def no_user_alpha(tweet):
            tweet_list = [ele for ele in tweet.split() if ele != 'user']
            clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split(
            ) if word.lower() not in eng_stopwords]
            return clean_mess
        no_punc_tweet = no_user_alpha(new_tweet)

        # Normalizing the words in tweets
        def normalization(tweet_list):
            lem = WordNetLemmatizer()
            normalized_tweet = []
            for word in tweet_list:
                normalized_text = lem.lemmatize(word, 'v')
                normalized_tweet.append(normalized_text)
            return normalized_tweet

        return normalization(no_punc_tweet)

    def text_processing_roman_urdu(self, tweet):
        tweet = p.clean(tweet)
        # Generating the list of words in the tweet (hastags and other punctuations removed)

        def form_sentence(tweet):
            tweet_blob = TextBlob(tweet)
            return ' '.join(tweet_blob.words)
        new_tweet = form_sentence(tweet)

        # Removing stopwords and words with unusual symbols
        def no_user_alpha(tweet):
            tweet_list = [ele for ele in tweet.split() if ele != 'user']
            clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split(
            ) if word.lower() not in rom_urdu_stopwords]
            return clean_mess
        no_punc_tweet = no_user_alpha(new_tweet)

        # Normalizing the words in tweets
        def normalization(tweet_list):
            lem = WordNetLemmatizer()
            normalized_tweet = []
            for word in tweet_list:
                normalized_text = lem.lemmatize(word, 'v')
                normalized_tweet.append(normalized_text)
            return normalized_tweet

        return normalization(no_punc_tweet)

    def text_processing_urdu(self, sentences):
        cleaned = []
        for sentence in sentences:
            text = re.sub(r"\d+", " ", str(sentence))
            # English punctuations
            text = re.sub(
                r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]+""", " ", text)
            # Urdu punctuations
            text = re.sub(r"[:؛؟’‘٭ء،۔]+", " ", text)
            # Arabic numbers
            text = re.sub(r"[٠‎١‎٢‎٣‎٤‎٥‎٦‎٧‎٨‎٩]+", " ", text)
            text = re.sub(r"[^\w\s]", " ", text)
            # Remove English characters and numbers.
            text = re.sub(r"[a-zA-z0-9]+", " ", text)
            # remove multiple spaces.
            text = re.sub(r" +", " ", text)
            text = text.split(" ")
            # some stupid empty tokens should be removed.
            text = [t.strip() for t in text if t.strip()]
            cleaned.append(" ".join(text))

        return cleaned

    def prediction(self, df: DataFrame) -> DataFrame:

        df['sentiment'] = loaded_model_MNB.predict(df["preprocessed"]) + loaded_model_LR.predict(df["preprocessed"]) + loaded_model_SVC.predict(df["preprocessed"]) + loaded_model_SGD.predict(df["preprocessed"])+ loaded_model_CNB.predict(df["preprocessed"])     
        df['sentiment'] = df['sentiment'].replace([1, 2, 3, 4, 5], [0, 0, 1, 1, 1])

        return df

    def predict_sentiment(self, df: DataFrame, language: str)-> DataFrame:
        if(language == 'en'):
            df['preprocessed'] = df.apply(self.preprocess_tweet, axis=1)
            df['preprocessed'] = df['preprocessed'].astype(
                str).apply(self.text_processing_english)
            df['preprocessed'] = [' '.join(map(str, l))
                                  for l in df['preprocessed']]
            df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        elif(language == 'ur'):
            df['preprocessed'] = self.text_processing_urdu(df.values)
        elif(language == 'in'):
            df['preprocessed'] = df.apply(self.preprocess_tweet, axis=1)
            df['preprocessed'] = df['preprocessed'].astype(
                str).apply(self.text_processing_roman_urdu)
            df['preprocessed'] = [' '.join(map(str, l))
                                  for l in df['preprocessed']]
            df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        else:
            print(
                'please choose one of these languages: "english: en", "urdu: ur" or "roman urdu: in"')
        return self.prediction(df)

#creating objects for both classes
text_sentiment = MethodsForText()
dataframe_sentiment = MethodsForDataframe()
