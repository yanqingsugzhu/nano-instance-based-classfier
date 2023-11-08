from nupack import *
import re
DFT_CONCERNTRATION = 1e-9
model = Model(material = 'dna', celsius = 21)
seqs_counter = 0
with open("similar_seqs-Copy1.txt", 'r') as seqs_reader:
    raw_seqs = seqs_reader.read().splitlines()
yields = []
while seqs_counter <= len(raw_seqs) -2:
    strand1 = Strand(raw_seqs[seqs_counter], name = "1")
    strand2 = Strand(raw_seqs[seqs_counter + 1], name = "2")
    if seqs_counter % 2 == 0:
        tube1 = Tube(strands={strand1 : DFT_CONCERNTRATION, strand2 : DFT_CONCERNTRATION}, complexes=SetSpec(max_size=2), name='tube1')
        my_result = tube_analysis([tube1], model=model,
        compute=['pfunc', 'pairs', 'mfe', 'sample', 'subopt'],
        options={'num_sample': 2, 'energy_gap': 0.5})
        reg = r"(1\+2)|(2\+1)"
        t1_result = my_result['tube1']
        for my_complex, conc in t1_result.complex_concentrations.items():
            is_matched = re.search(reg , my_complex.name)
            if not is_matched is None:
#                 print(conc/DFT_CONCERNTRATION)
                yields.append(conc/DFT_CONCERNTRATION)
    seqs_counter += 2
    print(seqs_counter)
with open("yields.txt", 'a+') as yields_writer:
    for _yield in yields:
        yields_writer.write(str(_yield)+'\n')
