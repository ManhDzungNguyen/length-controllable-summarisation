import re
import requests


PUNCTUATION = [":", ".", "!", "?"]


def truncate_text(text: str, max_len: int):
    if len(text) > max_len:
        text = text[:max_len]
        while text and text[-1] not in PUNCTUATION:
            text = text[:-1]

    return text


def is_contain_keyword_regex(text, reg_list):
    text = text.replace('_',' ').lower()
    if not reg_list:
        return True
    for reg in reg_list:
        search_res = re.search(reg,text,re.I)
        if search_res:
            return True
    return False


def normalize_text(text: str):
    text = re.sub(r"\s\s+", " ", text).strip()
    text = re.sub(r"\b[^ \t]*\.[^A-Z0-9\s]+", "", text)

    text = text.replace("🏻", "")
    text.replace(");this.closest('table').remove();", "")
    text = re.sub(
        "(Thứ .{2,4}|Chủ nhật),( ngày)? \d{1,2}\/\d{1,2}\/\d{4}( \d{1,2}:\d{1,2})?( AM| PM)?( \(GMT.{1,3}\))?",
        "",
        text,
    )

    text = re.sub("\(.*(Ảnh|Nguồn).*?\)", "", text)
    text = re.sub("\d{1,2} (giờ|phút) trước", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub("(\\r)*( )*(\\n)*( )*(\\r)*( )*(\\n)", ".", text)
    text = re.sub(r"\.( )*(\.)+", ". ", text)
    text = re.sub("\.(?!\d)", ". ", text)
    text = re.sub("(\.(\s)+)+", ". ", text)
    text = re.sub("<[^<]+?>", "", text)
    text = re.sub("\d{1,2}:\d{2}( )?\d{1,2}\/\d{1,2}\/\d{4}", "", text)
    text = re.sub("Ảnh(:)?(Getty)?", "", text)

    text = re.sub("\(ảnh:.*?\)", ".", text)
    text = re.sub("(\| )?(\(.{1,7}\)( )?)+$", "", text)
    text = re.sub(
        "\d{2} [\w]{3,4}, \d{4}. \d{2}.\d{2} (AM|PM) IST", "", text
    )  # 02 Mar, 2022, 10.01 AM IST
    text = re.sub("\d{2}\/\d{2}\/\d{4} \d{2}:\d{2} GMT(\+|-)\d{1,2}", "", text)
    # text = re.sub("\(.*(Photo|Ảnh|Images|Image|theo|nguồn).*\)", "", text, flags=re.I)
    text = re.sub("[A-Z].{5,10} , \d{2}:\d{2} (GMT(\+|-)\d{1,2})?", "", text)

    text = re.sub("^\d{1,10} minutes ago", "", text)
    text = re.sub("^\d{1,10} hours ago", "", text)
    text = re.sub("^\d{1,10} days ago", "", text)
    text = re.sub("^\d{1,10} years ago", "", text)
    text = re.sub("^\d{1,10} months ago", "", text)
    text = re.sub("^\d{1,10} minute ago", "", text)
    text = re.sub("^\d{1,10} day ago", "", text)
    text = re.sub("^\d{1,10} year ago", "", text)
    text = re.sub("^\d{1,10} month ago", "", text)
    text = re.sub("^\d{1,10} hour ago", "", text)
    text = re.sub("^(a|an) minute ago", "", text)
    text = re.sub("^(a|an) hour ago", "", text)
    text = re.sub("^(a|an) day ago", "", text)
    text = re.sub("^(a|an) month ago", "", text)
    text = re.sub("^(a|an) year ago", "", text)

    text = re.sub("\s+", " ", text)
    text = re.sub("Đọc chi tiết bài viết tại đây.*", "", text, flags=re.I)
    text = re.sub("(\d{1,2}:\d{2}( )*)\|( )*\d{1,2}(/|-)\d{2}(/|-)\d{4}", "", text)

    text = re.sub(
        "((chủ nhật)|(thứ bảy)|(thử sáu)|(thứ năm)|(thứ tư)|(thứ ba)|(thứ hai))([(\d)(\:)(,)(\|\/)(\s+)]+)((VOV)|(VTV))$",
        "",
        text,
        flags=re.I,
    )  # và Ukraine để giải quyết xung đột Chủ Nhật, 06:32, 20/03/2022 VOV.

    text = re.sub(
        "^((\d)|(\:)|(\.)|(\|)|(\s+)|(in bài biết)|(in bài viết)|(\/))+",
        "",
        text,
        flags=re.I,
    )  # 10:20 | 09/03/2022 In bài biết. 10:20 | 09/03/2022 In bài biết Việc xuất khẩu tôm sang thị trường Nga

    text = text.replace("|", "")
    text = re.sub("^.*?(Link nguồn)", "", text, flags=re.I)  # (

    text = re.sub("(a|p)\. m\.", "", text)
    text = re.sub("This article was last updated at \d{1,2}:\d{2} (GMT/UTC)?", "", text)

    text = re.sub("(\* )mời quý độc giả .{5,}", "", text, flags=re.I)
    text = re.sub(
        "Updated:.{3,10} \d{1,2}, \d{0,4} \d{1,2}:\d{2}:\d{2} (am|pm)?", "", text
    )

    text = re.sub("^vn.( )?}", "", text)
    text = re.sub("^AFP.", "", text)
    text = re.sub("AA.", "", text)
    text = re.sub("^AFP.", "", text)
    text = re.sub("Il y a \d{1,5} heures( \. )?", "", text)
    text = (
        text.replace("Xem tiếp>>>", "")
        .replace('"', '"')
        .replace("ชั่วโมงที่ผ่านมา .", "")
        .replace(
            "- Ảnh thời sự quốc tế - Chính trị-Quân sự - Thông tấn xã Việt Nam (TTXVN)",
            "",
        )
        .replace("TTXVN phát", "")
        .replace("| thefastnewz", "")
        .replace("| Swotverge", "")
        .replace("- VietBF", "")
        .replace("THÔNG LUẬN - ", "")
        .replace("vn. ", "")
        .replace("minh hoạ: Getty", "")
    )

    text = re.sub("/detail/[^\s]*", "", text)
    text = re.sub(
        "Điện thoại:.{1,80}$", "", text
    )  # Điện thoại:0242 248 7575, Hotline:0982 737679 Email:info@investglobal
    text = re.sub(
        "Use Next and Previous buttons to navigate (\d{1,2} )?(\d{1,2}of\d{1,2})?",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        "(Advertisement)? \d{1,5} (Story continues below)? (This advertisement has not loaded yet, but your article continues below.)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub("Photo by .{1,15}/Article content W", "", text, flags=re.I)
    text = re.sub("\(photo by .*?\)", "", text, flags=re.I)

    if text and text[-1] not in PUNCTUATION:
        text = text + "."

    return text
