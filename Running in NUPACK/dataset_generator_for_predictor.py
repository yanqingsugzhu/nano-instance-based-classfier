import random
from nupack import *
import re

NT_LENGTH = 59
HOM_THRE = 3
BASE_CHANGE_RATE = 0.5
DFT_CONCERNTRATION = 1e-9
SELF_COMP_THRE = 0.4


def gen_base_seq_pairs():
    bases = "ATCG"
    match_dict = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C'
    }
    pos_strand = ""
    neg_strand = ""
    last_base = 'B'
    repeat_counter = 0
    for i in range(NT_LENGTH):
        this_base = bases[random.randint(0, 3)]
        if this_base == last_base:
            if repeat_counter >= HOM_THRE - 2:
                while this_base == last_base:
                    this_base = bases[random.randint(0, 3)]
                repeat_counter = 0
            else:
                repeat_counter += 1
        else:
            repeat_counter = 0
        last_base = this_base
        pos_strand += this_base
    j = len(pos_strand) - 1
    while j > -1:
        neg_strand += match_dict[pos_strand[j]]
        j -= 1
    return pos_strand, neg_strand


def gen_similar_seqs(pos_strand, strands_num_to_gen):
    strands_counter = 0
    last_base = 'B'
    not_match_dict = {
        'A': ['T', 'C', 'G'],
        'T': ['A', 'C', 'G'],
        'C': ['A', 'T', 'G'],
        'G': ['A', 'T', 'C']
    }
    similar_seqs = []
    while strands_counter < strands_num_to_gen:
        similar_strand = ""
        for base in pos_strand:
            this_base = 'B'
            if random.randint(1, 10) < 10 * BASE_CHANGE_RATE:
                this_base = not_match_dict[base][random.randint(0, 2)]
                while this_base == last_base:
                    this_base = not_match_dict[base][random.randint(0, 2)]
            else:
                this_base = base
            similar_strand += this_base
            last_base = this_base
        strands_counter += 1
        similar_seqs.append(similar_strand)
    return similar_seqs


def analysis(seqs, reverse_comp):
    model = Model(material='dna', celsius=21)
    my_strands = [Strand(reverse_comp, name='-1')]
    my_con = {my_strands[0]: DFT_CONCERNTRATION * len(seqs) * 2}
    for i in range(0, len(seqs)):
        this_strand = Strand(seqs[i], name=str(i))
        my_strands.append(this_strand)
        my_con[this_strand] = DFT_CONCERNTRATION
        i += 1

    tube1 = Tube(strands=my_con, complexes=SetSpec(max_size=2), name='tube1')
    my_result = tube_analysis([tube1], model=model,
                              compute=['pfunc', 'pairs', 'mfe', 'sample', 'subopt'],
                              options={'num_sample': 2, 'energy_gap': 0.5})
    t1_result = my_result['tube1']  # equivalent to my_result['t1']
    strands_index = []
    strands_yields = []
    reg = r"((\-\d)|(\d))\+\-1"
    for my_complex, conc in t1_result.complex_concentrations.items():
        is_matched = re.search(reg, my_complex.name)
        if not is_matched is None:
            is_matched = is_matched.group()
            strands_index.append(int(is_matched[0: is_matched.index('+')]))
            strands_yields.append(conc / DFT_CONCERNTRATION)
    if len(strands_index) > 2:
        seqs_to_write = []
        this_yield = 0.0
        re_comp_pos = strands_index.index(-1)
        if strands_yields[re_comp_pos] > SELF_COMP_THRE:
            return False
        try:
            seqs_to_write.append(seqs[strands_index.index(0)])
        except ValueError:
            return False
        similar_seq_pos = 1
        control_flag = True
        while control_flag:
            try:
                control_flag = False
                strands_index.index(similar_seq_pos)
            except ValueError:
                similar_seq_pos += 1
                control_flag = True
        try:
            seqs_to_write.append(seqs[similar_seq_pos])
            this_yield = strands_yields[similar_seq_pos]
        except IndexError:
            return False
        with open('similar_seqs.txt', 'a+') as seqs_writer:
            for seq in seqs_to_write:
                seqs_writer.write(seq + '\n')
        with open('yields.txt', 'a+') as yields_writer:
            yields_writer.write(str(this_yield) + '\n')
        return True
    else:
        return False


def gen_high_yield_seq_pairs(pos_strand, neg_strand, steam_length, steam_num):
    steam_num_counter = 0
    miss_match_dict = {
        'A': ['A', 'C', 'G'],
        'T': ['T', 'C', 'G'],
        'C': ['A', 'T', 'C'],
        'G': ['A', 'T', 'G']
    }
    pos_strand = list(pos_strand)
    neg_strand = list(neg_strand)
    while (steam_num_counter < steam_num):
        ps_left_index = random.randint(0, len(pos_strand) - steam_length)
        neg_left_index = random.randint(0, len(pos_strand) - steam_length)
        steam_counter = 0
        ps_last_base = 'B'
        neg_last_base = 'B'
        try:
            ps_last_base = pos_strand[ps_left_index - 1]
        except IndexError:
            ps_last_base = 'B'
        try:
            neg_last_base = neg_strand[neg_left_index - 1]
        except IndexError:
            neg_last_base = 'B'
        while steam_counter < steam_length:
            ps_this_base = miss_match_dict[pos_strand[steam_counter + ps_left_index]][random.randint(0, 2)]
            while ps_this_base == ps_last_base:
                ps_this_base = miss_match_dict[pos_strand[steam_counter + ps_left_index]][random.randint(0, 2)]
            pos_strand[steam_counter + ps_left_index] = ps_this_base
            ps_last_base = ps_this_base
            neg_this_base = miss_match_dict[neg_strand[steam_counter + neg_left_index]][random.randint(0., 2)]
            while neg_this_base == neg_last_base:
                neg_this_base = miss_match_dict[neg_strand[steam_counter + neg_left_index]][random.randint(0., 2)]
            neg_strand[steam_counter + neg_left_index] = neg_this_base
            neg_last_base = neg_this_base
            steam_counter += 1
        steam_num_counter += 1
    return ''.join(pos_strand), ''.join(neg_strand)


def analysis_high_yield_seqs(pos_strand, neg_strand):
    strand1 = Strand(pos_strand, name="1")
    strand2 = Strand(neg_strand, name="2")
    model = Model(material='dna', celsius=21)
    tube1 = Tube(strands={strand1: DFT_CONCERNTRATION, strand2: DFT_CONCERNTRATION}, complexes=SetSpec(max_size=2),
                 name='tube1')
    my_result = tube_analysis([tube1], model=model,
                              compute=['pfunc', 'pairs', 'mfe', 'sample', 'subopt'],
                              options={'num_sample': 2, 'energy_gap': 0.5})
    reg = r"(1\+2)|(2\+1)"
    t1_result = my_result['tube1']
    for my_complex, conc in t1_result.complex_concentrations.items():
        is_matched = re.search(reg, my_complex.name)
        if not is_matched is None:
            return conc / DFT_CONCERNTRATION


high_yield_seqs_counter = 0
with open('59nt_trainning_seqs.txt', 'a+') as hy_s_writer:
    with open('59nt_trainning_yields.txt', 'a+') as hy_y_writer:
        while high_yield_seqs_counter < 50000:
            pos_strand, neg_strand = gen_base_seq_pairs()
            pos_strand, neg_strand = gen_high_yield_seq_pairs(pos_strand, neg_strand, random.randint(2, 8),
                                                              random.randint(3, 8))
            this_yield = analysis_high_yield_seqs(pos_strand, neg_strand)
            hy_s_writer.write(pos_strand + '\n')
            hy_s_writer.write(neg_strand + '\n')
            hy_y_writer.write(str(this_yield) + '\n')
            if high_yield_seqs_counter % 1000 == 0:
                print(high_yield_seqs_counter)
            high_yield_seqs_counter += 1