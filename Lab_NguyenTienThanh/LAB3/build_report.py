from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
D=Document(); sec=D.sections[0]
sec.top_margin=Inches(.62); sec.bottom_margin=Inches(.62); sec.left_margin=Inches(.7); sec.right_margin=Inches(.7)
styles=D.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(9.5); styles['Normal'].font.color.rgb=RGBColor(40,48,58)
for n,size,color in [('Title',24,'15324B'),('Heading 1',16,'15324B'),('Heading 2',12,'286A8A')]:
    st=styles[n]; st.font.name='Aptos Display'; st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor.from_string(color)

def para(text='',style=None,boldlead=None):
    p=D.add_paragraph(style=style)
    if boldlead and text.startswith(boldlead): p.add_run(boldlead).bold=True; p.add_run(text[len(boldlead):])
    else:p.add_run(text)
    return p

def shade(cell,color):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),color); tcPr.append(shd)

def table(headers,rows,widths=None):
    t=D.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Light Shading Accent 1'
    for c,h in zip(t.rows[0].cells,headers): c.text=str(h); shade(c,'15324B'); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for c in t.rows[0].cells:
        for r in c.paragraphs[0].runs: r.font.bold=True; r.font.color.rgb=RGBColor(255,255,255); r.font.size=Pt(8)
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row): c.text=str(v); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for c in cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(0)
                for r in p.runs:r.font.size=Pt(8)
    return t

def picture(name,width=3.1):
    p=D.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(ROOT/'results/figures'/name),width=Inches(width)); return p

def caption(t):
    p=D.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(t); r.italic=True; r.font.size=Pt(8); r.font.color.rgb=RGBColor(95,105,115)

p=D.add_paragraph(style='Title'); p.add_run('Lab 3 Nhận dạng tiếng nói với HMM')
p=para('Báo cáo thực hành | CSE457 - Xử lý âm thanh và tiếng nói'); p.paragraph_format.space_after=Pt(10)
para('Mục tiêu là xây dựng bộ nhận dạng 5 lệnh bằng Gaussian HMM trái-sang-phải, quan sát likelihood và Viterbi, rồi so sánh với DTW và thử mở rộng bằng lexicon cùng bigram. Kết quả trên bộ test sạch đạt 100% cho cả HMM lẫn DTW; kiểm tra nhiễu 20 dB giảm còn 60%, cho thấy độ nhạy môi trường rõ hơn mức accuracy sạch thể hiện.')
D.add_heading('1. Dữ liệu và quy trình',level=1)
para('Repository có bảy lệnh, mỗi lớp năm bản ghi. Bài dùng bat, dung, len, xuong, trai; files 01-03 là train và 04-05 là test, tổng cộng 15/10 utterance. Không có file nào xuất hiện ở cả hai split. Split được giữ nguyên giữa HMM và DTW. Dữ liệu hiện có ít hơn đề xuất 6 train + 2 test mỗi lớp và chỉ đại diện một điều kiện ghi âm.')
para('WAV được đọc mono 16 kHz, trim silence (top_db=30), pre-emphasis 0.97. Front-end dùng 13 MFCC với cửa sổ Hamming 25 ms, hop 10 ms, n_mels=26; delta và delta-delta được nối thành T×39. CMVN được tính riêng trong từng utterance. Hai heatmap dưới đây minh họa ba khối đặc trưng; sai phân bổ sung dấu hiệu chuyển động phổ theo thời gian.')
picture('features_bat.png',5.6); caption('Hình 1. MFCC, delta và delta-delta của một utterance bat.')
D.add_heading('2. Gaussian HMM và huấn luyện',level=1)
para('Mỗi lớp có một whole-word HMM. π=[1,0,...,0]; mỗi hàng chuyển tiếp chỉ cho phép self-loop và next-state, state cuối tự lặp. Xác suất phát xạ là Gaussian diagonal. Với ít utterance và 39 chiều, diagonal cần ít tham số hơn covariance đầy đủ (39 thay vì 39×40/2 phần tử hiệp phương sai mỗi state), nên giảm nguy cơ ước lượng không ổn định. Các transition bị cấm được giữ bằng 0 trong toàn bộ Baum-Welch.')
para('Mã NumPy/SciPy tự tính log-domain Forward-Backward, gamma_t(i) và xi_t(i,j), rồi cập nhật trung bình, phương sai và transition bằng EM. Likelihood Forward cộng xác suất của mọi đường state. Viterbi thay phép cộng bằng max, lưu backpointer và trả đường tốt nhất. Vì vậy Forward trả lời “chuỗi quan sát có khả năng dưới model bao nhiêu?”, còn Viterbi trả lời “đường state nào giải thích chuỗi tốt nhất?”.')
table(['Số state','Test accuracy','Vòng lặp EM TB'],[[x['n_states'],f"{x['accuracy']:.0%}",f"{x['mean_iterations']:.1f}"] for x in M['state_experiment']])
para('Cả 3, 5 và 7 state đều phân loại đúng 10/10 mẫu. Tập test quá nhỏ để phân biệt cấu hình hoặc kết luận 7 state tổng quát tốt hơn. Cấu hình báo cáo chính giữ 5 state theo case study. Baum-Welch có thể dừng ở cực đại cục bộ do khởi tạo, dữ liệu ít/mất cân bằng, quá nhiều state hoặc covariance suy biến; cần thử khởi tạo và giảm độ phức tạp khi convergence kém.')
D.add_page_break()
D.add_heading('3. Nhận dạng, score và alignment',level=1)
para('Quy tắc ML chọn lớp có log P(X|λ) cao nhất. Với cùng utterance, chia mọi score cho số frame không đổi thứ hạng; score/frame thuận tiện hơn khi đặt ngưỡng reject giữa utterance dài ngắn khác nhau. Bảng dưới là một test utterance và toàn bộ acoustic scores.')
sdf=pd.read_csv(ROOT/'results/hmm_test_scores.csv'); row=sdf[sdf.file=='bat_04.wav'].iloc[0]
rows=[[w,f"{row['score_'+w]:.1f}"] for w in M['labels']]
table(['Lớp model','log-likelihood'],rows)
para(f"bat_04 được nhận là {row['top1']} vì model này có likelihood cao nhất; top-1/top-2 cách nhau {row['margin']:.1f} log units trong trial này. Khoảng cách lớn phản ánh các lớp tách biệt mạnh trong tập nhỏ, không phải ngưỡng unknown đã được kiểm định.")
picture('hmm_confusion.png',3.0); picture('dtw_confusion.png',3.0); caption('Hình 2-3. Confusion matrices trên cùng 10 test utterance (trái HMM, phải DTW).')
para('Cả confusion matrix đều nằm trên đường chéo: 2/2 đúng ở mỗi lớp. Hai Viterbi path dưới đây không giảm theo frame, phù hợp topology left-to-right. Thời gian giữ một state là frame duration 10 ms x số frame liên tiếp; state là phân đoạn nội bộ của word HMM, không được xem như phoneme nếu không có giám sát phone.')
picture('viterbi_bat_04.png',3.0); picture('viterbi_xuong_04.png',3.0); caption('Hình 4-5. Alignment Viterbi cho bat_04 và xuong_04.')
D.add_heading('4. So sánh DTW và robustness',level=1)
table(['Phương pháp','Đặc trưng','Accuracy','Mẫu train/lớp','Thời gian test/trial'],[
 ['Gaussian HMM','39D MFCC động','100%','3',f"{M['mean_hmm_test_decode_seconds']*1000:.2f} ms"],
 ['DTW','13D MFCC tĩnh','100%','3 template',f"{M['mean_dtw_test_seconds']*1000:.2f} ms"]])
para('DTW và HMM ngang nhau về accuracy trong split nhỏ này. Đo thời gian tham khảo trên máy hiện tại; DTW so sánh đường trên ba template, còn HMM chấm một model/class, nên đây không phải benchmark chuẩn hóa. DTW cho warping path trực tiếp và đơn giản khi ít mẫu; HMM học phân bố trạng thái, có likelihood và mở rộng tự nhiên sang phone, lexicon và language model. HMM cần đủ dữ liệu để ước lượng tham số ổn định. Δ features có thể biểu diễn xu hướng phổ dù HMM vẫn mô hình chuỗi: emission frame độc lập có điều kiện theo state, còn delta đưa ngữ cảnh cục bộ vào vector quan sát.')
para(f"Static MFCC HMM cũng đạt {M['static_mfcc_hmm_accuracy']:.0%}; bộ dữ liệu chưa cho thấy delta cải thiện accuracy. Khi thêm white noise tổng hợp ở 20 dB SNR, accuracy còn {M['noise_20db_accuracy']:.0%} ({int(M['noise_20db_accuracy']*10)}/10). Đây là mismatch noise có kiểm soát, không phải đánh giá speaker mới. Full covariance không chạy: chỉ ba utterance train/lớp không đủ để ước lượng đáng tin cậy trong 39 chiều.")
D.add_page_break()
D.add_heading('5. Lexicon và bigram MAP re-ranking',level=1)
para('Whole-word HMM nhận trực tiếp một model cho mỗi từ nên không cần lexicon phát âm. Khi dùng phone/subword HMM, lexicon ánh xạ từ thành chuỗi phone; decoder có thể nối các phone HMM theo chuỗi này để tạo word network. `lexicon.txt` minh họa cú pháp WORD PHONE_1 ... PHONE_K và parser trong mã tạo dictionary. Các phone hiện là nhãn minh họa; cần thay bằng inventory và cách phát âm giảng viên cung cấp.')
para('Corpus lệnh ngắn trong `command_corpus.txt` dùng để đếm unigram/bigram. Add-k smoothing dùng P(w|prev)=(C(prev,w)+k)/(C(prev)+kV), k=0.5. Điểm re-ranking là log-likelihood/frame + λ log P(w|prev). Thử λ=0, 0.1, 0.3, 1 và thêm λ=100 để thấy prior mạnh. Với utterance có margin acoustic nhỏ nhất trong test, các candidate thật vẫn cách nhau lớn; λ≤1 không đổi quyết định. Corpus minh họa không đại diện ngôn ngữ tự nhiên, do đó LM ở đây chỉ trình diễn phép cộng Bayes, không chứng minh lợi ích nhận dạng.')
para('Ba điều làm EM hội tụ kém hoặc nghiệm xấu: khởi tạo state/emission không phù hợp, quá ít token so với số tham số, và số state/covariance quá phức tạp gây variance gần 0. Nhiễu/kênh khác và split không đúng cũng làm test giảm dù thuật toán hội tụ tốt.')
D.add_heading('6. Câu hỏi phân tích',level=1)
qa=[
('gamma_t và xi_t trong EM','gamma_t(i) là mức trách nhiệm của state i với frame t; xi_t(i,j) là mức trách nhiệm của chuyển i->j giữa hai frame. E-step tính hai đại lượng, M-step dùng chúng làm trọng số cập nhật.'),
('Vì sao diagonal covariance?','Giảm số tham số mạnh và tránh nghịch đảo covariance đầy đủ không ổn định khi dữ liệu nhỏ, feature dimension cao.'),
('Delta hữu ích ra sao?','Gói vận tốc và gia tốc biến thiên phổ trong vector frame; emission độc lập có điều kiện không tự biểu diễn tương quan cục bộ đó.'),
('Self-loop khi nói chậm','Cho phép ở lại state lâu hơn, nên HMM hấp thụ duration biến thiên mà không cần cố định số frame/state.'),
('Lexicon khi nào cần?','Không cần cho word HMM; cần khi từ được giải mã bằng phone/subword HMM để xác định chuỗi đơn vị phát âm hợp lệ.'),
('DTW path và Viterbi path','DTW căn chỉnh hai chuỗi feature bằng đường chi phí nhỏ; Viterbi tìm state sequence tốt nhất trong một HMM dựa trên xác suất chuyển và phát xạ.'),
('Unknown/reject','Argmax luôn có lớp thắng. Có thể từ chối khi log-likelihood/frame dưới ngưỡng hoặc margin top1-top2 nhỏ; phải hiệu chỉnh ngưỡng trên validation có unknown, không lấy test để chọn.'),
]
for q,a in qa:
    p=D.add_paragraph(); p.paragraph_format.space_after=Pt(4); r=p.add_run(q+'. '); r.bold=True; p.add_run(a)
D.add_heading('7. Kết luận và giới hạn',level=1)
para('Pipeline đã hoàn thành đủ phần trích đặc trưng, HMM, score/decoding, Viterbi, confusion matrix, DTW, thử số state, robustness, lexicon parser và bigram re-ranking. Accuracy sạch 100% cho cả HMM và DTW nhưng chỉ trên 10 test file; noise làm kết quả giảm 40 điểm phần trăm. Bước tiếp theo để kết luận có ý nghĩa là thu tối thiểu 6 train + 2 test mỗi lớp, thêm người nói/điều kiện ghi âm, và dùng phone inventory cùng corpus thật của lớp.')
p= D.add_paragraph(); p.paragraph_format.space_before=Pt(8); r=p.add_run('Tái lập: '); r.bold=True; p.add_run('chạy `python lab3.py` từ thư mục LAB3. Mã, notebook, split, model archive và bảng/biểu đồ được lưu cùng thư mục bài.')
D.core_properties.title='Lab 3 Nhận dạng tiếng nói với HMM'; D.core_properties.subject='CSE457 Speech Processing lab report'
D.save(ROOT/'Bao_cao_Lab3.docx')
