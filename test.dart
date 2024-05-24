import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

void main() async {
  
  final fileLoc = '000.jpg';

  final imgBytes = File(fileLoc).readAsBytesSync();

  final en64 = base64Encode(imgBytes);

  final url = 'http://127.0.0.1:8000/predict';

  final requestBody = jsonEncode({
    'image_data': en64,
    'top_n': 5
  });

  
  final response = await http.post(
    Uri.parse(url),
    headers: {'Content-Type': 'application/json'},
    body: requestBody,
  );

  
  if (response.statusCode == 200) {
    print('Response: ${response.body}');

  } else {
    print('Request failed with status: ${response.statusCode}');
    print('Response: ${response.body}');
  }
}
