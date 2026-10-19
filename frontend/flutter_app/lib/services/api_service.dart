import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class PredictionResult {
  final bool detected;
  final String riskLevel;      // "Low", "Medium", "High", "Unknown"
  final double confidence;
  final double probAggressive;
  final String? message;

  PredictionResult({
    required this.detected,
    required this.riskLevel,
    required this.confidence,
    required this.probAggressive,
    this.message,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      detected: json['detected'] ?? false,
      riskLevel: json['risk_level'] ?? 'Unknown',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      probAggressive: (json['prob_aggressive'] ?? 0.0).toDouble(),
      message: json['message'],
    );
  }
}

class ApiService {
  static String baseUrl = 'http://your-ngrok-url.ngrok.io'; // updated via Settings

  static Future<PredictionResult> predict(File imageFile) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      await http.MultipartFile.fromPath('image', imageFile.path),
    );

    try {
      final streamedResponse = await request.send()
          .timeout(const Duration(seconds: 15));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return PredictionResult.fromJson(json);
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
}