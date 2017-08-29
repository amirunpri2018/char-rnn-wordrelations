
import xml.sax.handler
import xml.sax
import sys

from collections import defaultdict

skipped_forms = set(['c', 'ci', 'cm', 'sms'])

class SaldoLexiconHandler(xml.sax.handler.ContentHandler):
    def __init__(self, result):
        self.result = result

    def startElement(self, name, attributes):
        if name == 'LexicalEntry':
            self.current = [ None, None, defaultdict(list) ]
        elif name == 'feat' and attributes['att'] == 'partOfSpeech':
            if self.current[1]:
                raise Exception("pos already set!")
            self.current[1] = attributes['val']
        elif name == 'feat' and attributes['att'] == 'writtenForm':
            if self.inFR:
                if self.current[0]:
                    raise Exception("lemma already set!")
                self.current[0] = attributes['val']
            else:
                self.wf = attributes['val']
        elif name == 'feat' and attributes['att'] == 'msd':            
            form_name = attributes['val']
            if self.wf[-1] != '-' and form_name not in skipped_forms:
                self.current[2][form_name].append(self.wf)
        elif name == 'FormRepresentation':
            self.inFR = True
                
    def endElement(self, name):
        if name == 'LexicalEntry':
            self.result.append(self.current)
        elif name == 'FormRepresentation':
            self.inFR = False

def read_saldom_lexicon(lexicon_file):
    parser = xml.sax.make_parser()
    result = []
    parser.setContentHandler(SaldoLexiconHandler(result))
    parser.parse(lexicon_file)
    return result

def to_form_str(forms, form):
    if form not in forms:
        return '-'
    else:
        return '/'.join(sorted(forms[form]))

def get_saldo_data(infile):
    relations = {}
    relations['n_plural']      = []
    relations['v_imperfect']   = []
    relations['v_progressive'] = []
    relations['v_presence']    = []
    relations['a_comparative'] = []
    relations['a_superlative'] = []
    relations['a_comparative_superlative'] = []

    lexicon_data = read_saldom_lexicon(infile)

    forms_for_pos = defaultdict(set)
    
    for lemma, pos, forms in lexicon_data:
        if pos in ['nn', 'av', 'vb']:
            forms_for_pos[pos].update(forms)
            
    forms_for_pos = { pos: sorted(forms) for pos, forms in forms_for_pos.items() }

    #print(forms_for_pos['av'])
    #print(forms_for_pos['vb'])
    #print(forms_for_pos['nn'])
    
    #advocera        vb      advocera advocera advoceras advocerar advoceras - - advocerandes advocerande advocerade advocerades - - advocerades advocerade advocerades advocerade advocerades advocerade advocerades advocerade advocerats advocerat advocerads advocerad advocerat advocerats
    #skivbroms       nn      skivbromsarnas skivbromsarna skivbromsars skivbromsar skivbromsens skivbromsen skivbroms skivbroms
    #klyvbar av      - klyvbarares klyvbarare klyvbaras klyvbara klyvbares klyvbare klyvbaras klyvbara klyvbaras klyvbara klyvbarts klyvbart klyvbars klyvbar klyvbarastes klyvbaraste klyvbarastes klyvbaraste klyvbarasts klyvbarast
    
    
    for lemma, pos, forms in lexicon_data:
        if pos in forms_for_pos:
            form_str = ' '.join(to_form_str(forms, f) for f in forms_for_pos[pos])
            #print(forms)
            #print('{}\t{}\t{}'.format(lemma, pos, form_str))
            if pos == 'nn':
              first = to_form_str(forms, forms_for_pos[pos][7]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][3]).split('/')[0]
              if first != '-' and second != '-':
                relations['n_plural'].append((first, second))

            elif pos == 'vb':
              first = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][9]).split('/')[0]
              if first != '-' and second != '-':
                relations['v_imperfect'].append((first, second))

              first = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][3]).split('/')[0]
              if first != '-' and second != '-':
                relations['v_presence'].append((first, second))

              first = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              if first != '-' and second != '-':
                relations['v_progressive'].append((first, second))

            elif pos == 'av':
              first = to_form_str(forms, forms_for_pos[pos][14]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              if first != '-' and second != '-':
                relations['a_comparative'].append((first, second))

              first = to_form_str(forms, forms_for_pos[pos][14]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][20]).split('/')[0]
              if first != '-' and second != '-':
                relations['a_superlative'].append((first, second))
              
              first = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][20]).split('/')[0]
              if first != '-' and second != '-':
                relations['a_comparative_superlative'].append((first, second))

    return relations

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == '-':
        infile = sys.stdin
    else:
        infile = sys.argv[1]
    lexicon_data = read_saldom_lexicon(infile)

    forms_for_pos = defaultdict(set)
    
    for lemma, pos, forms in lexicon_data:
        if pos in ['nn', 'av', 'vb']:
            forms_for_pos[pos].update(forms)
            
    forms_for_pos = { pos: sorted(forms) for pos, forms in forms_for_pos.items() }

    #print(forms_for_pos['av'])
    #print(forms_for_pos['vb'])
    #print(forms_for_pos['nn'])
    
    for lemma, pos, forms in lexicon_data:
        if pos in forms_for_pos:
            form_str = ' '.join(to_form_str(forms, f) for f in forms_for_pos[pos])
            print(forms)
            print('{}\t{}\t{}'.format(lemma, pos, form_str))




