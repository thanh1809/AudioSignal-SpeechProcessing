"""Build the Vietnamese Lab 5 report from the latest pipeline outputs."""
import csv, json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent; TAB=ROOT/'results'/'tables'; FIG=ROOT/'results'/'figures'; OUT=ROOT/'report.docx'
def rows(name):
    with (TAB/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)
def set_repeat(row):
    trPr=row._tr.get_or_add_trPr();el=OxmlElement('w:tblHeader');el.set(qn('w:val'),'true');trPr.append(el)
def set_cell(cell,text,bold=False,color=None,size=8):
    cell.text='';p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);r=p.add_run(str(text));r.bold=bold;r.font.size=Pt(size);r.font.name='Aptos'
    if color:r.font.color.rgb=RGBColor(*color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
def table(doc,headers,data,widths=None,size=8):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.style='Light Shading Accent 1';set_repeat(t.rows[0])
    for c,h in zip(t.rows[0].cells,headers):set_cell(c,h,True,(255,255,255),size);shade(c,'174A6E')
    for record in data:
        cells=t.add_row().cells
        for c,v in zip(cells,record):set_cell(c,v,size=size)
    if widths:
        for row in t.rows:
            for c,w in zip(row.cells,widths):c.width=Inches(w)
    return t
def para(doc,text='',style=None,boldlead=None):
    p=doc.add_paragraph(style=style);p.paragraph_format.space_after=Pt(5);p.paragraph_format.line_spacing=1.05
    if boldlead and text.startswith(boldlead):p.add_run(boldlead).bold=True;p.add_run(text[len(boldlead):])
    else:p.add_run(text)
    return p
def heading(doc,text,level=1):
    p=doc.add_paragraph(style=f'Heading {level}');p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(4);p.add_run(text);return p
def image(doc,name,width=6.25):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(1);p.add_run().add_picture(str(FIG/name),width=Inches(width))
def caption(doc,text):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(3);r=p.add_run(text);r.italic=True;r.font.size=Pt(8);r.font.color.rgb=RGBColor(85,99,110)
def page(doc):doc.add_page_break()

enh=rows('enhancement_snr.csv');acc=rows('speaker_condition_accuracy.csv');ica=rows('ica_conditions.csv');ver=json.loads((TAB/'verification_summary.json').read_text(encoding='utf-8'))
doc=Document();sec=doc.sections[0];sec.top_margin=Inches(.55);sec.bottom_margin=Inches(.55);sec.left_margin=Inches(.72);sec.right_margin=Inches(.72)
normal=doc.styles['Normal'];normal.font.name='Aptos';normal.font.size=Pt(9);normal.paragraph_format.space_after=Pt(4)
for sname,sz,col in [('Title',23,'174A6E'),('Heading 1',15,'174A6E'),('Heading 2',11,'25749B')]:
    s=doc.styles[sname];s.font.name='Aptos Display';s.font.size=Pt(sz);s.font.bold=True;s.font.color.rgb=RGBColor.from_string(col)
header=sec.header.paragraphs[0];header.alignment=WD_ALIGN_PARAGRAPH.RIGHT;r=header.add_run('CSE457  |  LAB 5');r.font.size=Pt(8);r.font.color.rgb=RGBColor(95,110,120)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=footer.add_run('Speech enhancement, speaker recognition and separation');r.font.size=Pt(8);r.font.color.rgb=RGBColor(100,110,120)

# Page 1
p=doc.add_paragraph(style='Title');p.add_run('Báo cáo Lab 5')
p=doc.add_paragraph();p.paragraph_format.space_after=Pt(12);r=p.add_run('Tăng cường tiếng nói, nhận dạng người nói và phân tách nguồn');r.bold=True;r.font.size=Pt(12);r.font.color.rgb=RGBColor(40,100,130)
para(doc,'Bài thực hành triển khai một pipeline có thể chạy lại từ đầu: tạo nhiễu cộng ở SNR đã định, khử nhiễu trong miền STFT, nhận dạng/xác minh người nói bằng MFCC và tách hỗn hợp hai microphone bằng FastICA. Kết quả cho thấy Wiener cải thiện SNR nhiều nhất trong cấu hình này; speaker-ID giảm trên tín hiệu nhiễu; ICA tách rất tốt hỗn hợp tức thời tổng hợp. Các con số speaker không đại diện cho người nói độc lập vì dữ liệu là giọng TTS.')
heading(doc,'Dữ liệu và quy trình')
para(doc,'Mỗi lớp có 4 utterance enrollment, 2 development và 2 test. spk01/spk02 là giọng Microsoft SAPI tổng hợp; spk03 là bản spk02 hạ pitch 4 bán âm để tạo lớp minh họa. Nhiễu quạt và nhiễu nền kiểu cafeteria được sinh xác định bằng seed. Hai nguồn phân tách là hai câu TTS khác nhau. Không dùng bản ghi người tham gia.')
table(doc,['Thành phần','Cấu hình'],[['Âm thanh','Mono, 16 kHz; PCM WAV'],['STFT','NFFT 512; Hann 400 mẫu (25 ms); hop 160 mẫu (10 ms)'],['Enhancement','SNR 0/5/10 dB; β 1/1.5/2; γ 0/0.01/0.05'],['Speaker','20 MFCC; thống kê mean/std; MFCC-only, delta và CMN ablation'],['Verification','θ quét trên dev; test giữ độc lập'],['ICA','FastICA 2 thành phần; ma trận mặc định và 3 điều kiện đối chứng']],[1.45,5.2],8)
heading(doc,'Giới hạn dữ liệu',2)
para(doc,'Tập speaker tổng hợp và spk03 có nguồn gốc từ spk02, do đó tập này không thỏa kiểm định trên ba người thật độc lập. FAR/FRR, accuracy và ablation chỉ mô tả pipeline trên các giọng tổng hợp này. Để đánh giá ngoài lớp học, cần thay dữ liệu bằng bản ghi có sự đồng ý, nhiều người độc lập và nhiều phiên thu.')

# Page 2
page(doc);heading(doc,'1. Tạo nhiễu và tăng cường tiếng nói')
para(doc,'Nhiễu được co giãn theo α = √(Pₛ/(Pᵥ·10^(R/10))) rồi cộng vào câu sạch. SNR đo lại lệch tối đa 0.0001 dB so với mục tiêu. Ước lượng PSD dùng chính noise-only clip đã nhân α để khớp mức nhiễu trong mixture. Spectral subtraction giữ pha noisy; Wiener dùng gain P̂ₛ/(P̂ₛ+P̂ᵥ).')
table(doc,['SNR vào','Đầu vào noisy','Spectral subtraction','Wiener'],[[f'{r} dB',f'{float(next(x for x in enh if x["target_snr_db"]==r and x["method"]=="noisy")["measured_snr_in_db"]):.2f}',f'{float(next(x for x in enh if x["target_snr_db"]==r and x["method"]=="spectral_subtraction")["snr_out_db"]):.2f}',f'{float(next(x for x in enh if x["target_snr_db"]==r and x["method"]=="wiener")["snr_out_db"]):.2f}'] for r in ('0','5','10')],[1.0,1.5,2.0,2.0],8)
caption(doc,'SNR đầu vào và SNR đầu ra (dB), đo so với câu sạch đã căn chỉnh cùng độ dài.')
image(doc,'enhancement_spectrograms.png',width=4.45);caption(doc,'Hình 1. Spectrogram câu sạch, noisy 5 dB và hai phương pháp tăng cường.')
para(doc,'Ở 5 dB, spectral subtraction đạt 9.95 dB (ΔSNR +4.95 dB), Wiener đạt 11.34 dB (+6.34 dB). Ở 0 dB, hai phương pháp đạt lần lượt 6.60 và 8.92 dB. β lớn hơn làm phép trừ mạnh hơn nhưng tăng nguy cơ méo phụ âm và musical noise; floor giữ lại năng lượng nhỏ, giảm các lỗ phổ sâu. Thí nghiệm đã quét đủ β và γ; bảng đầy đủ nằm trong `ss_parameter_sweep.csv`. Nhận xét nghe chủ quan cần được người nghe điền sau khi nghe A/B; báo cáo không gán nhãn nghe chưa thực hiện.')

# Page 3
page(doc);heading(doc,'2. Speaker identification')
para(doc,'Mỗi speaker model là trung bình bốn embedding enrollment đã L2-normalize. Embedding gồm mean/std MFCC và delta theo thời gian. Hai utterance test mỗi speaker tạo sáu lượt đánh giá cho từng điều kiện. Cosine score lớn nhất quyết định nhãn; confusion matrix dùng test sạch.')
table(doc,['Điều kiện test','Đúng / tổng','Accuracy'],[[r['condition'],f"{int(float(r['accuracy'])*int(r['n']))} / {r['n']}",f"{float(r['accuracy']):.1%}"] for r in acc],[2.6,1.7,1.5],8)
image(doc,'confusion_matrix.png',width=3.5);caption(doc,'Hình 2. Confusion matrix trên sáu utterance test sạch.')
para(doc,'Accuracy sạch là 100% (6/6), giảm còn 66.7% (4/6) ở noisy 5 dB và cũng 66.7% sau spectral subtraction hoặc Wiener. Trong mẫu rất nhỏ này, enhancement không phục hồi được toàn bộ phân biệt speaker và có thể làm mất một phần cue MFCC. Ablation MFCC-only/MFCC+delta/CMN được ghi trong `speaker_feature_ablation.csv`; do mỗi speaker chỉ có hai utterance test tổng hợp, chênh lệch accuracy ở đây không có sức mạnh suy luận.')
enroll=rows('enrollment_count_ablation.csv')
table(doc,['Số enrollment / speaker','Clean test accuracy'],[[r['enrollment_utterances_per_speaker'],f"{float(r['clean_test_accuracy']):.1%}"] for r in enroll],[3.2,2.4],8)
caption(doc,'Ablation số token enrollment (1/2/4) trên cùng sáu test utterance sạch.')

# Page 4
page(doc);heading(doc,'3. Speaker verification')
para(doc,'Genuine trials so sánh test utterance với model cùng speaker; impostor trials so sánh với hai model khác. Ngưỡng được chọn bằng điểm dev có |FAR−FRR| nhỏ nhất rồi khóa trước khi tính test. Với θ = %.3f, dev FAR/FRR là %.1f%% / %.1f%%; test FAR/FRR là %.1f%% / %.1f%% trên %d genuine và %d impostor trials.'%(ver['threshold'],100*ver['dev_far'],100*ver['dev_frr'],100*ver['test_far'],100*ver['test_frr'],ver['n_test_genuine'],ver['n_test_impostor']))
image(doc,'far_frr.png',width=5.8);caption(doc,'Hình 3. FAR và FRR theo ngưỡng cosine; đường dọc là ngưỡng chọn trên dev.')
heading(doc,'Diễn giải',2)
para(doc,'Test FAR 33.3% cho thấy một phần impostor bị chấp nhận ở ngưỡng đã chọn, còn FRR 0% trên sáu lượt genuine. Kết quả này không phải điểm vận hành tin cậy: số lượt nhỏ, ngưỡng cosine gần 1, và spk03 được tạo từ spk02. Nếu chọn ngưỡng bằng test, số đo sẽ thiên lệch lạc quan; việc tách dev/test ngăn rò rỉ đó nhưng không thay thế dữ liệu đủ lớn.')
heading(doc,'Khác biệt giữa các tác vụ',2)
para(doc,'Identification chọn một nhãn trong tập speaker đã đăng ký và thường báo accuracy/confusion matrix. Verification kiểm tra một claim có phải đúng speaker hay không và báo FAR/FRR theo ngưỡng. Speaker change detection tìm thời điểm đổi người nói; source separation khôi phục đồng thời các tín hiệu chồng lấn, là bài toán khác.')

# Page 5
page(doc);heading(doc,'4. Hai microphone và FastICA')
para(doc,'Hai câu TTS khác người được RMS-normalize và trộn theo x = A s, với A mặc định [[1, 0.6], [0.4, 1]], det(A)=0.76. FastICA chỉ nhận hai kênh mixture, không nhận A. Sau khi tách, Hungarian assignment trên |ρ| xử lý hoán vị và đảo dấu; WAV được peak-normalize để nghe.')
chosen=next(r for r in ica if r['condition']=='default')
para(doc,'Ở cấu hình mặc định, ma trận |ρ| trước ghép là [[%.4f, %.4f], [%.4f, %.4f]]; tương quan ghép là %.5f và %.5f. Ma trận thay thế cho trung bình |ρ| %.5f; thêm nhiễu Gaussian σ=0.03 cho %.5f. Với A gần suy biến (det=0.19), trung bình |ρ| là %.5f.'%(float(chosen['corr_s1_e1']),float(chosen['corr_s1_e2']),float(chosen['corr_s2_e1']),float(chosen['corr_s2_e2']),float(chosen['matched_corr_s1']),float(chosen['matched_corr_s2']),float(next(r for r in ica if r['condition']=='alternate')['mean_matched_abs_corr']),float(next(r for r in ica if r['condition']=='noisy')['mean_matched_abs_corr']),float(next(r for r in ica if r['condition']=='near_singular')['mean_matched_abs_corr'])))
image(doc,'ica_correlation.png',width=2.8);caption(doc,'Hình 4. Tương quan tuyệt đối trước khi xử lý hoán vị ICA.')
image(doc,'ica_waveforms_spectrograms.png',width=4.35);caption(doc,'Hình 5. Nguồn, hai mixture microphone và estimate đã ghép theo tương quan.')
para(doc,'ICA tách gần hoàn hảo trong thí nghiệm tổng hợp vì nguồn có đủ khác biệt thống kê và mô hình trộn tức thời chính xác. det(A) nhỏ làm nghịch đảo nhạy với sai số dù FastICA vẫn cho điểm cao trong lượt chạy này. Phòng thật có trễ, vang và trộn convolutive; do đó kết quả này không dự đoán chất lượng tách trong phòng.')

# Page 6
page(doc);heading(doc,'5. Trả lời câu hỏi và kết luận')
answers=[
'Spectral subtraction sửa magnitude vì phase noisy dễ dùng trực tiếp và phép trừ power đơn giản. Ở SNR thấp, phase noisy chứa nhiều nhiễu nên tái tạo có thể méo dù magnitude tốt hơn.',
'β là mức over-subtraction: tăng β thường giảm residual noise nhưng có thể xóa thành phần speech. γ là spectral floor, ngăn magnitude rơi về zero và giảm lỗ phổ/musical noise, đổi lại để lọt thêm noise.',
'Wiener gain gần 1 ở bin speech-dominant nên giữ bin đó; gần 0 ở bin noise-dominant nên suy giảm bin đó. Ước lượng PSD sai làm gain lệch.',
'Căn chỉnh thời gian bảo đảm mỗi mẫu clean trừ đúng mẫu enhanced. Nếu estimate trễ 10 ms, sai số tăng mạnh và SNR đo được thấp giả tạo.',
'Identification chọn một speaker trong tập lớp và đánh giá bằng accuracy/confusion matrix. Verification chấp nhận/từ chối claim, đánh giá FAR/FRR theo threshold.',
'Chọn threshold trên test làm rò rỉ thông tin nhãn test vào quyết định và khiến FAR/FRR lạc quan. Tăng ngưỡng thường giảm FAR nhưng tăng FRR.',
'MFCC phản ánh phổ âm học, nhưng phổ cũng chịu ảnh hưởng của đường phát âm, formant, pitch và thói quen phát âm nên có thông tin speaker.',
'CMN có thể giảm lệch phổ do kênh, nhưng baseline thống kê MFCC cũng có thể dùng mean utterance làm cue; trừ mean có thể loại một phần cue này.',
'ICA chỉ xác định nguồn tới hoán vị và scale/sign. Vì vậy thứ tự output, biên độ và dấu tuyệt đối không được xác định từ mixture đơn thuần.',
'Nếu hai hàng A gần tỉ lệ nhau, microphone thu gần như cùng mixture; det(A) tiến về 0, nghịch đảo trở nên suy biến và tách nhạy với nhiễu.',
'Trộn tức thời là mỗi kênh tổ hợp tuyến tính cùng thời điểm. Phòng vang tạo đáp ứng xung khác nhau theo micro và trộn convolutive theo thời gian.',
'Trong mẫu này accuracy giảm từ 100% sạch xuống 66.7% khi noisy và vẫn 66.7% sau enhancement. Do dữ liệu TTS rất nhỏ, không kết luận enhancement luôn giúp/hại trên người thật.'
]
for i in range(0,12,2):
    t=doc.add_table(rows=1,cols=2);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for j in range(2):
        idx=i+j;c=t.rows[0].cells[j];c.width=Inches(3.25);c.text='';p=c.paragraphs[0];p.paragraph_format.space_after=Pt(5);p.paragraph_format.line_spacing=1.0
        r=p.add_run(f'{idx+1}. ');r.bold=True;r.font.color.rgb=RGBColor(23,74,110);r.font.size=Pt(8)
        r=p.add_run(answers[idx]);r.font.size=Pt(8)
heading(doc,'Kết luận kỹ thuật',2)
para(doc,'(1) Ở SNR vào 5 dB, SNR ra tăng +4.95 dB với spectral subtraction và +6.34 dB với Wiener. (2) Speaker-ID tổng hợp giảm 33.3 điểm phần trăm khi thêm nhiễu 5 dB; enhancement trong lượt này không đưa accuracy về mức sạch. (3) FastICA đạt |ρ| ghép trung bình 0.99994 cho ma trận mặc định, nhưng đây là trộn tổng hợp tức thời. Kết quả được tái tạo bằng `python lab5.py`; bảng, WAV và hình lưu trong `results/`.')

doc.core_properties.title='Báo cáo Lab 5 - Tăng cường và nhận dạng tiếng nói'
doc.core_properties.subject='Speech enhancement, speaker recognition and separation'
doc.core_properties.author='CSE457 Lab 5'
doc.save(OUT)
print(OUT)
