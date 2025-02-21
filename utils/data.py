def dataf():
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
    ]
    json_file_name = 'google.json'
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
    gc = gspread.authorize(credentials)
    spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1rj3nwKG1bn6gr4T2hCNEU9ycnENQaat3UbF9KL-PfG8/edit?usp=sharing'
    doc = gc.open_by_url(spreadsheet_url)
    worksheet = doc.worksheet('test1')

    worksheet.update_acell('E1', 'E1 updated')

    cell_data = worksheet.acell('A1').value
    return cell_data

#https://ysyblog.tistory.com/354
#https://sojinhwan0207.tistory.com/200
#requirements.txt 수정