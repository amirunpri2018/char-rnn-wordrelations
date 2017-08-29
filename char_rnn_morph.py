#!/usr/bin/python

'''
Copyright (C) 2017 Olof Mogren

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, random, time, math, urllib3, random, os, argparse, sys
from urllib.request import urlopen
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datareader_saldo import get_saldo_data
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

default_datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

parser = argparse.ArgumentParser()
parser.add_argument("--max_iterations", type=int, default=10000, help="Max iterations")
parser.add_argument("--batch_size", type=int, default=22, help="batch size")
parser.add_argument("--rnn_depth", type=int, default=2, help="RNN depth")
parser.add_argument("--hidden_size", type=int, default=350, help="RNN hidden size")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
parser.add_argument("--disable_attention", action="store_true", help="Disable the attention mechanism.")
parser.add_argument("--disable_classification", action="store_true", help="Disable the relation classification loss.")
parser.add_argument("--super_attention", action="store_true", help="Ensable the super attention mechanism.")
parser.add_argument("--l2_regularization", type=float, default=0.00005, help="L2 regularization factor.")
parser.add_argument("--save_dir", type=str, default=None, help="Path to directory where the model parameters can be saved and loaded.")
parser.add_argument("--data_dir", type=str, default=default_datadir, help="Path to directory where the dataset can be saved and loaded.")
parser.add_argument("--languages", type=str, default='english', help="Comma-separated list of languages. Implemented: english.")
parser.add_argument("--test_only", action="store_true", help="Disables the training procedure. Tries to load model and then test the best_iteration.")
parser.add_argument("--keep_probability", type=float, default=1.0, help="Dropout keep probability.")
args = parser.parse_args()

random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)


data_url_english = 'https://raw.githubusercontent.com/en-wl/wordlist/master/alt12dicts/2of12id.txt'
data_url_swedish = 'https://svn.spraakdata.gu.se/sb-arkiv/pub/lmf/saldom/saldom.xml'

# In the spell-checker dataset, the longest word is 24 characters.
# Most words are significantly shorter.
max_sequence_len = 30
use_cuda = False

teacher_forcing_ratio = .5

num_valid_examples_per_relation = 200
num_test_examples_per_relation  = 200

use_cuda = torch.cuda.is_available()

sorted_relation_labels           = sorted(['n_plural','v_imperfect','v_progressive','v_presence','a_comparative','a_superlative','a_comparative_superlative'])
relations                        = {}
vocab                            = []
vocab_size                       = -1
reverse_vocab                    = {}
num_relation_classes                 = None
all_letters                      = string.ascii_letters + " .,;'-ÅåÄäÖöÜüßßßßß"

def prepare_data():
  global relations,num_relation_classes
  all_relations                    = {}
  for language in args.languages.split(','):
    all_relations[language]      = {}
    for relation in sorted_relation_labels:
      all_relations[language][relation]      = []
  
  # each relation is counted twice, as they can be reversed.
  num_relation_classes = len(sorted_relation_labels)*2*len(args.languages.split(','))

  partitions = ['train','validation','test']
  for partition in partitions:
    relations[partition] = {}
    for language in ['english','swedish']:
      relations[partition][language] = {}
      for key in sorted_relation_labels:
        relations[partition][language][key]      = []

  exists = True
  for partition in relations:
    for language in args.languages.split(','):
      for key in sorted_relation_labels:
        if os.path.exists(os.path.join(args.data_dir, '{}/{}/{}.txt'.format(partition,language,key))):
          with open(os.path.join(args.data_dir, '{}/{}/{}.txt'.format(partition,language,key)), 'r') as f:
            for line in f:
              words = line.split()
              #print(words)
              relations[partition][language][key].append((words[0], words[1]))
          all_relations[language][key] += relations[partition][language][key]
        else:
          exists = False
          break
  if exists:
    for language in args.languages.split(','):
      #print(all_relations[language])
      for key in all_relations[language]:
        #print(all_relations[language][key])
        #for partition in relations:
        #  print(relations[partition][language][key])
        print('From filesystem: {}: {}: total: {}, num_train: {}, num_valid: {}, num_test: {}'.format(language, key, len(all_relations[language][key]), len(relations['train'][language][key]), len(relations['validation'][language][key]), len(relations['test'][language][key])))
  else:
    print('Data will be downloaded')
    all_relations = {}
    for language in args.languages.split(','):
      all_relations[language] = {}
      for relation in sorted_relation_labels: 
        all_relations[language][relation]      = []
    language = 'english'
    with urlopen(data_url_english) as f_web:
      print('reading {}'.format(data_url_english))
      for line in f_web:
        #print(line)
        words = [x.decode('utf-8') for x in line.split()]
        #print(words)
        for i in range(len(words)-1,-1,-1):
          if words[i][0] == '~' or words[i][0] == '+' or words[i][0] == '-' or words[i][0] == '!':
            words[i] = words[i][1:]
            if words[i] == '':
              del words[i]
        first = words[0]
        POS = words[1]
        for i in range(len(words)-1,1,-1):
          if words[i][0] == '(' or words[i][-1] ==')':
            del words[i]
          elif words[i][0] == '{' or words[i][-1] =='}':
            del words[i]
          elif words[i] == '|' or words[i] =='/':
            if i+1 < len(words):
              del words[i+1]
            del words[i]
        if POS == 'A:':
          # we don't seem to have inflections of these.
          #print('adjective')
          if len(words) != 4:
            print('Unexpected length of adjective line: {}, {}. Ignoring.'.format(line, words))
            continue
          second = words[2]
          if len(second) == 0:
            continue
          all_relations[language]['a_comparative'].append((first,second))
          third = words[3]
          if len(third) == 0:
            continue
          all_relations[language]['a_superlative'].append((first,third))
          all_relations[language]['a_comparative_superlative'].append((second,third))
        elif POS == 'N:':
          # most of the time, these have only one inflection, the plural.
          # Sometimes, there is an alternative form in parentheses.
          # Currently, we discard these (see for loop above).
          # TODO: test to just include them?
          #print('noun')
          if len(words) != 3:
            print('Unexpected length of noun line: {}, {}. Ignoring.'.format(line, words))
            continue
          second = words[2]
          all_relations[language]['n_plural'].append((first,second))
        elif POS == 'V:':
          #print('verb')
          if len(words) != 5 and len(words) != 6:
            print('Unexpected length of verb line: {}, {}. Ignoring.'.format(line, words))
            continue
          second = words[2]
          all_relations[language]['v_imperfect'].append((first,second))
          # sometimes, there is an extra form in second place. Sometimes not. Indexing from end.
          third = words[-2]
          all_relations[language]['v_progressive'].append((first,third))
          fourth = words[-1]
          all_relations[language]['v_presence'].append((first,fourth))

    for key in all_relations[language]:
      random.shuffle(all_relations[language][key])
      relations['train'][language][key]      = all_relations[language][key][num_test_examples_per_relation+num_valid_examples_per_relation:]
      relations['validation'][language][key] = all_relations[language][key][num_test_examples_per_relation:num_test_examples_per_relation+num_valid_examples_per_relation]
      relations['test'][language][key]    = all_relations[language][key][:num_test_examples_per_relation]
      for partition in relations:
        try: os.makedirs(os.path.join(os.path.join(args.data_dir, partition), language))
        except: pass
        with open(os.path.join(args.data_dir, '{}/{}/{}.txt'.format(partition,language,key)), 'w') as f:
          for (first, second) in relations[partition][language][key]:
            f.write('{} {}\n'.format(first, second))
      print('{}: {}: total: {}, num_train: {}, num_valid: {}, num_test: {}'.format(language, key, len(all_relations[language][key]), len(relations['train'][language][key]), len(relations['validation'][language][key]), len(relations['test'][language][key])))


  # TODO: infile=https://svn.spraakdata.gu.se/sb-arkiv/pub/lmf/saldom/saldom.xml, get_saldo_data(infile)
    language = 'swedish'
    saldo_filename = os.path.join(args.data_dir, 'saldom.xml')
    if os.path.exists(saldo_filename):
      print('found {} on disk.'.format(data_url_swedish))
    else:
      print('downloading {}'.format(data_url_swedish))
      with urlopen(data_url_swedish) as f_web:
        with open(saldo_filename , 'w') as f:
          for line in f_web:
            f.write(line.decode('utf-8')+'\n')
    swerel = get_saldo_data(saldo_filename)
    for key in swerel:
      random.shuffle(swerel[key])
      relations['train'][language][key]      = swerel[key][num_test_examples_per_relation+num_valid_examples_per_relation:]
      relations['validation'][language][key] = swerel[key][num_test_examples_per_relation:num_test_examples_per_relation+num_valid_examples_per_relation]
      relations['test'][language][key]    = swerel[key][:num_test_examples_per_relation]
      for partition in relations:
        try: os.makedirs(os.path.join(os.path.join(args.data_dir, partition), language))
        except: pass
        with open(os.path.join(args.data_dir, '{}/{}/{}.txt'.format(partition,language,key)), 'w') as f:
          for (first, second) in relations[partition][language][key]:
            f.write('{} {}\n'.format(first, second))
      print('{}: {}: total: {}, num_train: {}, num_valid: {}, num_test: {}'.format(language, key, len(swerel[key]), len(relations['train'][language][key]), len(relations['validation'][language][key]), len(relations['test'][language][key])))
    #for partition in swerel:
    #  relations[partition][language] = swerel[partition]

def print_len_stats():
  maxlen = 0
  minlen = 100
  lens = {}
  for p in relations:
    for l in relations:
      for k in relations[l]:
        for w1,w2 in relations[p][l][k]:
          lens[len(w1)] = lens.get(len(w1), 0)+1
          lens[len(w2)] = lens.get(len(w2), 0)+1
          maxlen = max(maxlen, len(w1))
          maxlen = max(maxlen, len(w2))
          minlen = min(minlen, len(w1))
          minlen = min(minlen, len(w2))
  print('wordlens: min {}, max{}.'.format(minlen, maxlen))
  l = list(lens.keys())
  l.sort()
  for ln in l:
    print('len: {}, num: {}.'.format(ln, lens[ln]))

def initialize_vocab():
  global vocab, vocab_size, reverse_vocab
  vocab.append('<PAD>')
  vocab.append('<BOS>')
  vocab.append('<EOS>')
  vocab.append('<UNK>')
  for i in range(len(all_letters)):
    vocab.append(all_letters[i])
  for i in range(len(vocab)):
    reverse_vocab[vocab[i]] = i
  vocab_size = len(vocab)
  
# turn a unicode string to plain ascii

def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters
  )

#print(unicodeToAscii('Ślusàrski'))

def line_to_index_tensor(lines, pad_before=True, append_bos_eos=False):
  seqlen = 0
  for l in lines:
    if len(l) > seqlen:
      seqlen = len(l)
  seqlen = min(seqlen, max_sequence_len)
  if append_bos_eos:
    seqlen += 2
  tensor = torch.zeros(len(lines), seqlen).long()
  tensor += reverse_vocab['<PAD>']
  for b in range(len(lines)):
    begin_pos = 0
    if pad_before:
      begin_pos = max(0,seqlen-len(lines[b]))
    else:
      begin_pos = 0
    if append_bos_eos:
      begin_pos += 1
      tensor[b][begin_pos-1] = reverse_vocab['<BOS>']
    for li, letter in enumerate(lines[b]):
      idx = li+begin_pos
      if idx >= seqlen:
        break
      tensor[b][idx] = reverse_vocab.get(letter, reverse_vocab['<UNK>'])
    if append_bos_eos:
      eos_pos = min(seqlen-1,begin_pos+len(lines[b]))
      tensor[b][eos_pos] = reverse_vocab['<EOS>']
  if use_cuda:
    tensor = tensor.cuda()
  return tensor

class AttentionModule(nn.Module):
  def __init__(self, hidden_size):
    super(AttentionModule, self).__init__()
    self.hidden_size    = hidden_size
    self.sigmoid        = nn.Sigmoid()
    self.tanh           = nn.Tanh()
    self.attn           = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
    self.attn2          = nn.Linear(self.hidden_size * 2, 1)
    self.attn_combine   = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.softmax        = nn.Softmax()
    if use_cuda:
      self.sigmoid.cuda()
      self.tanh.cuda()
      self.attn.cuda()
      self.attn2.cuda()
      self.attn_combine.cuda()
      self.softmax.cuda()

  def forward(self, hidden, decoder_out, encoder_states):
    #attention mechanism:
    # hidden is shape [depth, batch, hidden_size].
    # We use only the top level hidden state: [batch, hidden_size]
    attention_weights = []
    for i in range(encoder_states.size()[0]):
      attention_weights.append(self.attn2(self.tanh(self.attn(torch.cat((hidden, encoder_states[i]), 1)))))
    attention_weights=torch.stack(attention_weights, dim=1).squeeze()
    #print(attention_weights.size())
    attention_weights = self.softmax(attention_weights)
#print(attn_weights.size())
    #print(hidden.size())
    #print(decoder_out.size())
    #print(encoder_states.size())
    attention_weights = attention_weights.view(hidden.size()[0],1,-1)[:,:,:encoder_states.size()[0]]
    encoder_states_batchfirst = encoder_states.permute(1,0,2)
    attention_applied = torch.bmm(attention_weights,encoder_states_batchfirst)

    attention_applied = attention_applied.view(hidden.size()[0], -1)
    #input_emb    = input_emb.view(args.batch_size, -1)
    #print(decoder_out.size())
    #print(attention_applied.size())
    attention_out = torch.cat((decoder_out, attention_applied), 1)
    attention_out = self.attn_combine(attention_out)
    attention_out = attention_out.view(hidden.size()[0], -1)
    attention_out = self.tanh(attention_out)
    return attention_out, attention_applied

class RnnRelationModel(nn.Module):
  def __init__(self, vocab_size, num_relation_classes, hidden_size, depth, disable_attention, super_attention):
    super(RnnRelationModel, self).__init__()

    self.hidden_size       = hidden_size
    self.num_relation_classes  = num_relation_classes
    self.depth             = depth
    self.disable_attention = disable_attention
    self.super_attention   = super_attention

    self.embedding      = nn.Embedding(vocab_size, hidden_size)

    self.rnn_demo1      = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=depth)
    self.rnn_demo2      = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=depth)
    self.rnn_query1     = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=depth)

    self.fc_rel         = nn.Linear(hidden_size*2, hidden_size)
    self.rel_out        = nn.Linear(hidden_size, num_relation_classes)
    self.fc_combined    = nn.Linear(hidden_size*2, hidden_size*depth)
    self.tanh           = nn.Tanh()

    # Zero initial state:
    self.hidden_initial = Variable(torch.zeros(depth, 1, self.hidden_size))

    self.rnn_decoder    = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=depth)
    self.linear         = nn.Linear(hidden_size, vocab_size)
    self.logsoftmax     = nn.LogSoftmax()

    self.dropout        = nn.Dropout(p=(1.0-args.keep_probability))

    if not disable_attention:
      #self.sigmoid        = nn.Sigmoid()
      #self.relu           = nn.ReLU()
      #self.attn           = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
      #self.attn2          = nn.Linear(self.hidden_size * 2, 1)
      #self.attn_combine   = nn.Linear(self.hidden_size * 2, self.hidden_size)
      #self.softmax        = nn.Softmax()
      self.attention_query = AttentionModule(hidden_size)
      if self.super_attention:
        #self.attention_demo1 = AttentionModule(hidden_size)
        #self.attention_demo2 = AttentionModule(hidden_size)
        self.attention_meta  = AttentionModule(hidden_size)

    if use_cuda:
      self.embedding.cuda()
      self.rnn_demo1.cuda()
      self.rnn_demo2.cuda()
      self.rnn_query1.cuda()
      self.rnn_decoder.cuda()
      self.fc_rel.cuda()
      self.rel_out.cuda()
      self.fc_combined.cuda()
      self.tanh.cuda()
      self.hidden_initial = self.hidden_initial.cuda()
      self.linear.cuda()
      self.logsoftmax.cuda()
      self.dropout.cuda()
      if not disable_attention:
        self.attention_query.cuda()

  # input = demo1
  def forward(self, input, input2, query1, query2, teacher_forcing_r=0.0, disable_dropout=False):
    h_init = self.hidden_initial.repeat(1, input.size()[0], 1)
    #print('{}, {}, {}'.format(input.size(), input2.size(), query1.size()))
    demo1_emb = self.embedding(input).permute(1,0,2)
    #print(input.size())
    #print(self.hidden_initial.size())
    #print(h_init.size())
    #demo1_emb = demo1_emb.contiguous()
    demo1_o, demo1_h = self.rnn_demo1(demo1_emb, h_init)
    demo2_emb = self.embedding(input2).permute(1,0,2)
    demo2_o, demo2_h = self.rnn_demo2(demo2_emb, h_init)
    query1_emb = self.embedding(query1).permute(1,0,2)
    query1_o, query1_h = self.rnn_query1(query1_emb, h_init)
    #print(query1[0].size())
    rels_out = torch.cat((demo1_o[-1], demo2_o[-1]), 1)#, query1_o[-1]), 1)
    if not disable_dropout:
      rels_out = self.dropout(rels_out)
    #print(intermediate.size())
    rel_encoder_out = self.tanh(self.fc_rel(rels_out)) #.clamp(min=0)
    if not disable_dropout:
      rel_encoder_out = self.dropout(rel_encoder_out)
    relation_classification_head = self.logsoftmax(self.rel_out(rel_encoder_out))
    
    encoders_out = torch.cat((rel_encoder_out, query1_o[-1]), 1)
    encoder_out = self.tanh(self.fc_combined(encoders_out)) #.clamp(min=0)
    if not disable_dropout:
      encoder_out = self.dropout(encoder_out)

    use_teacher_forcing = True if random.random() < teacher_forcing_r else False
    
    hidden = torch.stack(torch.chunk(encoder_out, chunks=self.depth, dim=1), dim=0)

    output_classes = []
    choices = [] #only for debugging
    #print(hidden.size())
    if use_teacher_forcing:
      #print('teacher forcing')
      # contiguous is needed for the view below. Only once per batch.
      dec_in_emb_seq = self.embedding(query2).permute(1,0,2).contiguous()
      # add sequence dimension with len 1:
      input_emb = dec_in_emb_seq[0].view(1, -1, self.hidden_size)
    else:
      #print('not teacher forcing')
      dec_in = Variable(torch.zeros(input.size()[0], 1).long()+reverse_vocab['<BOS>'])
      dec_in = dec_in.cuda() if use_cuda else dec_in
      input_emb = self.embedding(dec_in).permute(1,0,2).contiguous()
    for i in range(max_sequence_len):
      #print(hidden[0,0,:30])
      # We use the GRU as a (deep) GRUCell, one step at a time:
      #print(input_emb.size())
      #print(hidden.size())
      outputs, hidden = self.rnn_decoder(input_emb, hidden)
      if self.disable_attention:
        output = outputs[-1]
      else:
        # hidden from top cell, last (only) output, and decoder_outputs.
        output, weighted_sum = self.attention_query(hidden[-1], outputs[-1], query1_o)
        if self.super_attention:
          output, _ = self.attention_meta(hidden[-1], outputs[-1], torch.stack([weighted_sum, demo2_o[-1]], dim=0))
          
      #print(output.size())
      output_classes.append(self.logsoftmax(self.linear(output)))
      topv, topi = output_classes[-1].data.topk(1)
      choices.append(topi[0][0])
      #print(output_classes[-1])
      #print(len(output_classes))
      #print(topi.size())
      #print(topi[0][0])
      if use_teacher_forcing:
        if i+1 < dec_in_emb_seq.size()[0]:
          # add sequence dimension with len 1:
          input_emb = dec_in_emb_seq[i+1].view(1, input.size()[0], self.hidden_size)
          #print('teacher: input_emb size: {}'.format(input_emb.size()))
      else:
        if use_cuda:
          dec_in = Variable(torch.cuda.LongTensor(topi))
        else:
          dec_in = Variable(torch.LongTensor(topi))
        input_emb = self.embedding(dec_in).permute(1,0,2)
        #print('no teacher: input_emb size: {}'.format(input_emb.size()))
      #print(input)
    #print(choices)
    #print('{} teacher forcing: {}'.format('   ' if use_teacher_forcing else 'not', ''.join([vocab[c] for c in choices if (c != reverse_vocab['<PAD>'] and c != reverse_vocab['<BOS>'] and c != reverse_vocab['<EOS>'])])))

    output_classes = torch.stack(output_classes, dim=0)

    return output_classes, relation_classification_head


#def categoryFromOutput(output):
#  top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
#  category_i = top_i[0][0]
#  return all_categories[category_i], category_i

#print(categoryFromOutput(output))

def randomChoice(l):
  return l[random.randint(0, len(l) - 1)]

def get_pair_of_pairs(partition='train', language=None, relation=None, position=None, reverse=None, batch_size=None):
  #each argument specified as None, will result in a random choice.
  #print(list(all_relations.keys()))
  dw1s = []
  dw2s = []
  qw1s = []
  qw2s = []
  if batch_size == None:
    batch_size = args.batch_size
  relation_class = torch.zeros(batch_size).long()
  for b in range(batch_size):
    if language is None:
      language_choice = randomChoice(args.languages.split(','))
    else:
      language_choice = language
    if relation is None:
      relation_type = randomChoice(sorted_relation_labels)
    else:
      relation_type = relation
    language_index = args.languages.split(',').index(language_choice)
    relation_class[b] = language_index*len(sorted_relation_labels)*2+sorted_relation_labels.index(relation_type)*2
    if position is None:
      demo_word1, demo_word2 = randomChoice(relations[partition][language_choice][relation_type])
      query_word1, query_word2 = randomChoice(relations[partition][language_choice][relation_type])
    else:
      # will use consecutive pairs from the already shuffled list:
      demo_word1, demo_word2 = relations[partition][language_choice][relation_type][position+b]
      query_word1, query_word2 = relations[partition][language_choice][relation_type][position+b+1]
    if reverse is None:
      reverse_choice = random.choice([True, False])
    else:
      reverse_choice = reverse
    if reverse_choice:
      # reversing the relation.
      # plus one for reverse.
      relation_class[b] = sorted_relation_labels.index(relation_type)*2+1
      tmp = demo_word2
      demo_word2 = demo_word1
      demo_word1 = tmp
      tmp = query_word2
      query_word2 = query_word1
      query_word1 = tmp
    dw1s.append(demo_word1)
    dw2s.append(demo_word2)
    qw1s.append(query_word1)
    qw2s.append(query_word2)
  dw1t = Variable(line_to_index_tensor(dw1s, pad_before=True))
  dw2t = Variable(line_to_index_tensor(dw2s, pad_before=True))
  qw1t = Variable(line_to_index_tensor(qw1s, pad_before=True))
  qw2t = Variable(line_to_index_tensor(qw2s, pad_before=False, append_bos_eos=True))
  relation_class = Variable(relation_class)
  return dw1t, dw2t, qw1t, qw2t, relation_class


def train(demo_word1, demo_word2, query_word1, query_word2, relation_class, model, criterion, class_criterion, optimizer):
  optimizer.zero_grad()

  outputs, relation_classification_output = model(demo_word1, demo_word2, query_word1, query_word2, teacher_forcing_ratio, disable_dropout=False)

  # Remove 'BOS'!
  # Also, go from shape [batch, seqlen] -> [seqlen,batch] to be compatible with output tensor.
  target = query_word2.permute(1,0)[1:,:]
  loss = 0.0
  pad_target = Variable(torch.zeros(outputs.size()[1]).long()+reverse_vocab['<PAD>'])
  if use_cuda:
    pad_target = pad_target.cuda()
  for i in range(outputs.size()[0]):
    if i < len(target):
      t = target[i]
    else:
      t = pad_target
    loss += criterion(outputs[i], t)
  loss = loss / outputs.size()[0]

  total_loss = loss
  rel_loss_val = 0.0
  if relation_class is not None and not args.disable_classification:
    relation_classification_loss = class_criterion(relation_classification_output, relation_class)
    total_loss += relation_classification_loss
    rel_loss_val = relation_classification_loss.data[0]

  total_loss.backward()
  optimizer.step()

  return outputs, loss.data[0], rel_loss_val

def predict(demo_word1, demo_word2, query_word1, query_word2, relation_class, model, criterion, class_criterion):
  outputs, relation_classification_output = model(demo_word1, demo_word2, query_word1, query_word2, teacher_forcing_r=0.0, disable_dropout=True)

  # Remove 'BOS'!
  # Also, go from shape [batch, seqlen] -> [seqlen,batch] to be compatible with output tensor.
  target = query_word2.permute(1,0)[1:,:]
  loss = 0.0
  pad_target = Variable(torch.zeros(outputs.size()[1]).long()+reverse_vocab['<PAD>'])
  if use_cuda:
    pad_target = pad_target.cuda()
  for i in range(outputs.size()[0]):
    if i < len(target):
      t = target[i]
    else:
      t = pad_target
    loss += criterion(outputs[i], t)
  loss = loss / outputs.size()[0]
  
  rel_loss_val = 0.0
  if relation_class is not None and not args.disable_classification:
    relation_classification_loss = class_criterion(relation_classification_output, relation_class)
    rel_loss_val = relation_classification_loss.data[0]
    
  return outputs, loss.data[0], rel_loss_val


def topi(x):
  topv, topi = x.data.topk(1)
  return topi

def to_scalar(var):
  # returns a python float
  return var.view(-1)[0]
  #return var.view(-1).data.tolist()[0]

def word_tensor_to_string(t, handle_special_tokens=False):
  word = ''
  for o in range(t.size()[0]):
    index = to_scalar(t[o].data)
    if index == reverse_vocab['<PAD>']:
      continue
    if handle_special_tokens:
      if index == reverse_vocab['<EOS>']:
        break
      elif index == reverse_vocab['<BOS>']:
        continue
    word += vocab[index]
  return word

def prediction_tensor_to_string(t):
  word = ''
  for o in t:
    index = to_scalar(topi(o))
    if index == reverse_vocab['<EOS>']:
      break
    elif index == reverse_vocab['<PAD>']:
      continue
    elif index == reverse_vocab['<BOS>']:
      continue
    word += vocab[index]
  return word

def evaluate(partition, model, criterion, class_criterion):
  evaluation_steps = 0
  loss_sum = 0.0
  counts = {}
  counts['total'] = 0
  counts_exact_correct = {}
  counts_exact_correct['total'] = 0
  for language in args.languages.split(','):
    #print(language)
    counts[language[:2]+'_total'] = 0
    counts_exact_correct[language[:2]+'_total'] = 0
    for relation in sorted_relation_labels:
      counts[language[:2]+'_'+relation] = 0
      counts[language[:2]+'_'+relation+'_r'] = 0
      counts_exact_correct[language[:2]+'_'+relation] = 0
      counts_exact_correct[language[:2]+'_'+relation+'_r'] = 0
      evaluation_pos = 0
      evaluation_batch_size = 10
      #print('valpos: {}, limit: {}'.format(evaluation_pos ,num_valid_examples_per_relation-evaluation_batch_size*2))
      while evaluation_pos < num_valid_examples_per_relation-evaluation_batch_size*2:
        #print('evaluation iteration {}'.format(evaluation_pos))
        for reverse in [False,True]:
          demo_word1, demo_word2, query_word1, query_word2, relation_class = get_pair_of_pairs(partition=partition, language=language, relation=relation, position=evaluation_pos, reverse=reverse, batch_size=evaluation_batch_size)
          evaluation_pos += evaluation_batch_size*2
          if use_cuda:
            relation_class = relation_class.cuda()
          #print('predict')
          val_outputs, val_loss, val_class_loss = predict(demo_word1, demo_word2, query_word1, query_word2, relation_class, model, criterion, class_criterion )
          #print('done predicting')
          loss_sum += val_loss
          evaluation_steps += 1
          relation_label = language[:2]+'_'+relation
          if reverse:
            relation_label += '_r'
          for b in range(query_word2.size()[0]):
            prediction = prediction_tensor_to_string(val_outputs.permute(1,0,2)[b])
            correct    = word_tensor_to_string(query_word2[b], handle_special_tokens=True)
            #print('{}: prediction: {}, correct: {}'.format(b, prediction, correct))
            if prediction == correct:
              #print('correct!')
              counts_exact_correct['total'] += 1
              counts_exact_correct[language[:2]+'_total'] += 1
              counts_exact_correct[relation_label] += 1
            #else:
            #  print('not correct!')
            counts['total'] += 1
            counts[language[:2]+'_total'] += 1
            counts[relation_label] += 1
      
    relation1  = word_tensor_to_string(demo_word1[0])
    relation2  = word_tensor_to_string(demo_word2[0])
    query      = word_tensor_to_string(query_word1[0])
    prediction = prediction_tensor_to_string(val_outputs.permute(1,0,2)[0])
    correct    = word_tensor_to_string(query_word2[0], handle_special_tokens=True)
    print('sample relation ({} set): ({}-{}), query: ({}-{}), prediction: "{}"'.format(partition, relation1, relation2, query, correct, prediction))
  return loss_sum, evaluation_steps, counts_exact_correct, counts

def main():
  n_iters = args.max_iterations
  print_every =  100 # 5000
  plot_every = 200
  save_every = 200
  begin_iteration = 0
  current_accuracy = -1.0
  best_accuracy = -1.0
  best_iteration = 0

  print('languages: {}'.format(args.languages.split(',')))

  prepare_data()
  initialize_vocab()
  #print(num_relation_classes)

  model = RnnRelationModel(vocab_size, num_relation_classes, args.hidden_size, args.rnn_depth, args.disable_attention, args.super_attention)

  if args.save_dir:
    print('save_dir: {}'.format(args.save_dir))
    if os.path.exists(os.path.join(args.save_dir, 'current-iteration.txt')):
      with open(os.path.join(args.save_dir, 'current-iteration.txt'), 'r') as f:
        for l in f:
          if l:
            print('iteration: {}'.format(l))
            begin_iteration = int(l)
            begin_iteration += 1
          else:
            print('empty line!')
      with open(os.path.join(args.save_dir, 'best-iteration.txt'), 'r') as f:
        for l in f:
          if l:
            best_iteration = int(l)
      with open(os.path.join(args.save_dir, 'best-accuracy.txt'), 'r') as f:
        for l in f:
          if l:
            best_accuracy = float(l)

    last_iteration = begin_iteration - 1
    if os.path.exists(os.path.join(args.save_dir, 'parameters-{}.torch'.format(last_iteration))):
      print('Loading model parameters from {}.'.format(args.save_dir))
      model.load_state_dict(torch.load(os.path.join(args.save_dir, 'parameters-{}.torch'.format(last_iteration))))
    else:
      if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'progress.data'), 'a') as f:
          header = '#iteration training_loss validation_loss'
          keys = ['total']
          for l in args.languages.split(','):
            keys.append(l[:2]+'_total')
          for k in sorted_relation_labels:
            for l in args.languages.split(','):
              keys.append(l[:2]+'_'+k)
              keys.append(l[:2]+'_'+k+'_r')
          for k in keys:
            header += ' val:{}'.format(k)
          f.write(header+'\n')
      print('No model parameters found at location: {}. Will proceed with freshly initialized parameters and try to save to this location.'.format(args.save_dir))
  else:
    print('Created model with fresh parameters.')

  if use_cuda:
    model = model.cuda()

  # Keep track of losses for plotting
  current_loss = 0
  all_losses = []

  def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

  start = time.time()

  loss_weights = torch.ones(vocab_size)
  #loss_weights[reverse_vocab['<PAD>']] = 0.0
  if use_cuda:
    loss_weights = loss_weights.cuda()
  criterion = nn.NLLLoss(weight=loss_weights)
  class_criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.l2_regularization)

  iter=begin_iteration-1
  if not args.test_only:
    for iter in range(begin_iteration, n_iters + 1):
      # get random example:
      demo_word1, demo_word2, query_word1, query_word2, relation_class = get_pair_of_pairs('train')
      if use_cuda:
        relation_class = relation_class.cuda()
      outputs, loss, class_loss = train(demo_word1, demo_word2, query_word1, query_word2, relation_class, model, criterion, class_criterion, optimizer)
      current_loss += loss

      # Print iter number, loss, name and prediction
      if iter % print_every == 0:
        relation1  = word_tensor_to_string(demo_word1[0])
        relation2  = word_tensor_to_string(demo_word2[0])
        query      = word_tensor_to_string(query_word1[0])
        prediction = prediction_tensor_to_string(outputs.permute(1,0,2)[0])
        correct    = word_tensor_to_string(query_word2[0], handle_special_tokens=True)
        print('train: step {} ({}%, {}) loss: {:.4f} classification loss {:.4f}'.format(iter, iter / n_iters * 100, timeSince(start), loss, class_loss))
        print('sample relation (training set): ({}-{}), query: ({}-{}), prediction: "{}"'.format(relation1, relation2, query, correct, prediction))
        sys.stdout.flush()
        
        dw1l = ['äpple', 'banan', 'see', 'babian']
        dw1 = Variable(line_to_index_tensor(dw1l, pad_before=True))
        dw2l = ['äpplen', 'bananer', 'saw', 'babianer']
        dw2 = Variable(line_to_index_tensor(dw2l, pad_before=True))
        qw1l = ['nyckel', 'päron', 'eat', 'chips']
        qw1 = Variable(line_to_index_tensor(qw1l, pad_before=True))
        qw2l = ['nycklar', 'päron', 'ate', 'chips']
        qw2 = Variable(line_to_index_tensor(qw2l, pad_before=False, append_bos_eos=True))

        relation_class = None
        outputs, _, _ = predict(dw1, dw2, qw1, qw2, relation_class, model, criterion, class_criterion )
        for b in range(len(dw1l)):
          prediction = prediction_tensor_to_string(outputs.permute(1,0,2)[b])
          print('{} is to {} as {} is to {}.'.format(dw1l[b], dw2l[b], qw1l[b], prediction))

        val_loss_sum, validation_steps, counts_exact_correct, counts = evaluate('validation', model, criterion, class_criterion)
        print('validation loss: {:.4f}, accuracy (exact match): {:.4f}. Total validation tests: {}'.format(val_loss_sum/validation_steps, counts_exact_correct['total']/counts['total'], counts['total']))
        current_accuracy = counts_exact_correct['total']/counts['total']
        #print(counts_exact_correct)
        #print(counts)
        sys.stdout.flush()
        

        
        if args.save_dir:
          with open(os.path.join(args.save_dir, 'progress.data'), 'a') as f:
            line = '{} {} {}'.format(iter, loss, val_loss_sum/validation_steps)
            keys = ['total']
            for l in args.languages.split(','):
              keys.append(l[:2]+'_total')
            for k in sorted_relation_labels:
              for l in args.languages.split(','):
                keys.append(l[:2]+'_'+k)
                keys.append(l[:2]+'_'+k+'_r')
            for k in keys:
              line += ' {}'.format(counts_exact_correct[k]/counts[k])
            f.write(line+'\n')
            print(line)
      # Add current loss avg to list of losses
      if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
      if iter % save_every == 0:
        if args.save_dir:
          torch.save(model.state_dict(), os.path.join(args.save_dir, 'parameters-{}.torch'.format(iter)))
          with open(os.path.join(args.save_dir, 'current-iteration.txt'), 'w') as f:
            f.write(str(iter)+'\n')
          if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_iteration = iter
            with open(os.path.join(args.save_dir, 'best-iteration.txt'), 'w') as f:
              f.write(str(iter)+'\n')
            with open(os.path.join(args.save_dir, 'best-accuracy.txt'), 'w') as f:
              f.write(str(current_accuracy)+'\n')
  
  # time for test!
  if os.path.exists(os.path.join(args.save_dir, 'parameters-{}.torch'.format(best_iteration))):
    print('Loading model parameters from {}.'.format(args.save_dir))
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'parameters-{}.torch'.format(best_iteration))))
    test_loss_sum, test_steps, counts_exact_correct, counts = evaluate('test', model, criterion, class_criterion)
    with open(os.path.join(args.save_dir, 'test.data'), 'a') as f:
      header = '#iteration test_loss'
      line = '{} {}'.format(best_iteration, test_loss_sum/test_steps)
      keys = ['total']
      for l in args.languages.split(','):
        keys.append(l[:2]+'_total')
      for k in sorted_relation_labels:
        for l in args.languages.split(','):
          keys.append(l[:2]+'_'+k)
          keys.append(l[:2]+'_'+k+'_r')
      for k in keys:
        header += ' {}'.format(k)
        line += ' {}'.format(counts_exact_correct[k]/counts[k])
      f.write(header+'\n')
      f.write(line+'\n')
      print(line)
  else:
    print('Error: Tried to load the best model (from iteration: {}), but it was not found at \'{}\'.'.format(best_iteration, os.path.join(args.save_dir, 'parameters-{}.torch'.format(best_iteration))))


if __name__ == '__main__':
  main()
