import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const CanineCueApp());
}

class CanineCueApp extends StatelessWidget {
  const CanineCueApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'CanineCue',
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Arial',
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();

  // Backend address
  static const String backendUrl = 'http://127.0.0.1:8000';

  bool _isLoading = false;

  String? _selectedImageName;
  String? _selectedVideoName;

  String? _prediction;
  double? _confidence;

  int? _aggressiveFrames;
  int? _calmFrames;
  int? _analyzedFrames;

  double? _aggressivePercentage;
  double? _durationSeconds;

  // =========================================================
  // IMAGE UPLOAD AND DETECTION
  // =========================================================

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
      );

      if (image == null) {
        return;
      }

      setState(() {
        _selectedImageName = image.name;
        _selectedVideoName = null;

        _prediction = null;
        _confidence = null;

        _aggressiveFrames = null;
        _calmFrames = null;
        _analyzedFrames = null;
        _aggressivePercentage = null;
        _durationSeconds = null;

        _isLoading = true;
      });

      final bytes = await image.readAsBytes();

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/scan'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          bytes,
          filename: image.name,
        ),
      );

      final response = await request.send();

      final responseBody =
          await response.stream.bytesToString();

      if (response.statusCode != 200) {
        throw Exception(responseBody);
      }

      final data = jsonDecode(responseBody);

      if (!mounted) {
        return;
      }

      setState(() {
        _prediction = data['prediction'];

        _confidence =
            (data['confidence'] as num).toDouble();

        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _isLoading = false;
      });

      _showError(
        'Could not connect to the AI server.',
      );
    }
  }

  // =========================================================
  // VIDEO UPLOAD AND DETECTION
  // =========================================================

  Future<void> _pickVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.gallery,
      );

      if (video == null) {
        return;
      }

      setState(() {
        _selectedVideoName = video.name;
        _selectedImageName = null;

        _prediction = null;
        _confidence = null;

        _aggressiveFrames = null;
        _calmFrames = null;
        _analyzedFrames = null;
        _aggressivePercentage = null;
        _durationSeconds = null;

        _isLoading = true;
      });

      final bytes = await video.readAsBytes();

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/scan/video'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          bytes,
          filename: video.name,
        ),
      );

      final response = await request.send();

      final responseBody =
          await response.stream.bytesToString();

      if (response.statusCode != 200) {
        throw Exception(responseBody);
      }

      final data = jsonDecode(responseBody);

      if (!mounted) {
        return;
      }

      setState(() {
        _prediction = data['prediction'];

        _confidence =
            (data['confidence'] as num).toDouble();

        _aggressiveFrames =
            (data['aggressive_frames'] as num).toInt();

        _calmFrames =
            (data['calm_frames'] as num).toInt();

        _analyzedFrames =
            (data['analyzed_frames'] as num).toInt();

        _aggressivePercentage =
            (data['aggressive_percentage'] as num)
                .toDouble();

        _durationSeconds =
            (data['duration_seconds'] as num)
                .toDouble();

        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _isLoading = false;
      });

      // Show the real error so we can identify the problem.
      _showError(
        'Video error: $e',
      );
    }
  }

  // =========================================================
  // ERROR MESSAGE
  // =========================================================

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
      ),
    );
  }

  // =========================================================
  // RESULT CARD
  // =========================================================

  Widget _buildResultCard() {
    if (_isLoading) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(30),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
        ),
        child: const Column(
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 20),
            Text(
              'Analyzing...',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'Please wait while CanineCue analyzes the input.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.grey,
              ),
            ),
          ],
        ),
      );
    }

    if (_prediction == null) {
      return const SizedBox.shrink();
    }

    final bool aggressive =
        _prediction == 'aggressive';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        color: aggressive
            ? Colors.red.shade50
            : Colors.green.shade50,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: aggressive
              ? Colors.red.shade300
              : Colors.green.shade300,
        ),
      ),
      child: Column(
        children: [
          Icon(
            aggressive
                ? Icons.warning_rounded
                : Icons.check_circle_rounded,
            size: 55,
            color: aggressive
                ? Colors.red
                : Colors.green,
          ),

          const SizedBox(height: 12),

          Text(
            aggressive ? 'AGGRESSIVE' : 'CALM',
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: aggressive
                  ? Colors.red
                  : Colors.green,
            ),
          ),

          const SizedBox(height: 8),

          Text(
            'Confidence: '
            '${((_confidence ?? 0) * 100).toStringAsFixed(2)}%',
            style: const TextStyle(
              fontSize: 17,
            ),
          ),

          // -------------------------------------------------
          // Video information
          // -------------------------------------------------

          if (_selectedVideoName != null) ...[
            const SizedBox(height: 20),

            const Divider(),

            const SizedBox(height: 15),

            _ResultRow(
              label: 'Video',
              value: _selectedVideoName!,
            ),

            _ResultRow(
              label: 'Duration',
              value:
                  '${(_durationSeconds ?? 0).toStringAsFixed(2)} seconds',
            ),

            _ResultRow(
              label: 'Frames analyzed',
              value: '${_analyzedFrames ?? 0}',
            ),

            _ResultRow(
              label: 'Aggressive frames',
              value: '${_aggressiveFrames ?? 0}',
            ),

            _ResultRow(
              label: 'Calm frames',
              value: '${_calmFrames ?? 0}',
            ),

            _ResultRow(
              label: 'Aggressive percentage',
              value:
                  '${(_aggressivePercentage ?? 0).toStringAsFixed(2)}%',
            ),
          ],
        ],
      ),
    );
  }

  // =========================================================
  // MAIN UI
  // =========================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FB),

      // -----------------------------------------------------
      // APP BAR
      // -----------------------------------------------------

      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,

        title: const Row(
          children: [
            Icon(
              Icons.pets,
              color: Colors.indigo,
              size: 30,
            ),
            SizedBox(width: 10),
            Text(
              'CanineCue',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 22,
              ),
            ),
          ],
        ),

        actions: [
          TextButton(
            onPressed: () {},
            child: const Text('Home'),
          ),

          TextButton(
            onPressed: () {},
            child: const Text('History'),
          ),

          TextButton(
            onPressed: () {},
            child: const Text('About'),
          ),

          const SizedBox(width: 15),
        ],
      ),

      // -----------------------------------------------------
      // BODY
      // -----------------------------------------------------

      body: SingleChildScrollView(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(
              maxWidth: 1100,
            ),

            child: Padding(
              padding: const EdgeInsets.all(30),

              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,

                children: [
                  const SizedBox(height: 20),

                  const Text(
                    'Dog Aggression Detection',
                    style: TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1F2937),
                    ),
                  ),

                  const SizedBox(height: 10),

                  const Text(
                    'Detect and analyze aggressive behavior '
                    'in dogs using AI-powered computer vision.',
                    style: TextStyle(
                      fontSize: 17,
                      color: Colors.grey,
                    ),
                  ),

                  const SizedBox(height: 35),

                  // =================================================
                  // DETECTION CARD
                  // =================================================

                  Container(
                    width: double.infinity,

                    padding: const EdgeInsets.all(30),

                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius:
                          BorderRadius.circular(20),

                      boxShadow: [
                        BoxShadow(
                          blurRadius: 15,
                          spreadRadius: 2,
                          color:
                              Colors.black.withOpacity(0.06),
                        ),
                      ],
                    ),

                    child: Column(
                      children: [
                        const Icon(
                          Icons.pets,
                          size: 70,
                          color: Colors.indigo,
                        ),

                        const SizedBox(height: 20),

                        const Text(
                          'Start Detection',
                          style: TextStyle(
                            fontSize: 26,
                            fontWeight: FontWeight.bold,
                          ),
                        ),

                        const SizedBox(height: 10),

                        const Text(
                          'Choose an option below to analyze '
                          'dog behavior.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 15,
                            color: Colors.grey,
                          ),
                        ),

                        const SizedBox(height: 30),

                        // =================================================
                        // UPLOAD IMAGE BUTTON
                        // =================================================

                        SizedBox(
                          width: 260,
                          height: 55,

                          child: ElevatedButton.icon(
                            onPressed:
                                _isLoading
                                    ? null
                                    : _pickImage,

                            icon: const Icon(
                              Icons.image,
                            ),

                            label: const Text(
                              'Upload Image',
                              style: TextStyle(
                                fontSize: 16,
                              ),
                            ),

                            style:
                                ElevatedButton.styleFrom(
                              backgroundColor:
                                  Colors.indigo,

                              foregroundColor:
                                  Colors.white,

                              shape:
                                  RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius.circular(
                                  12,
                                ),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 15),

                        // =================================================
                        // UPLOAD VIDEO BUTTON
                        // =================================================

                        SizedBox(
                          width: 260,
                          height: 55,

                          child: OutlinedButton.icon(
                            onPressed:
                                _isLoading
                                    ? null
                                    : _pickVideo,

                            icon: const Icon(
                              Icons.video_library,
                            ),

                            label: const Text(
                              'Upload Video',
                              style: TextStyle(
                                fontSize: 16,
                              ),
                            ),

                            style:
                                OutlinedButton.styleFrom(
                              foregroundColor:
                                  Colors.indigo,

                              side:
                                  const BorderSide(
                                color: Colors.indigo,
                              ),

                              shape:
                                  RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius.circular(
                                  12,
                                ),
                              ),
                            ),
                          ),
                        ),

                        // =================================================
                        // SELECTED IMAGE
                        // =================================================

                        if (_selectedImageName != null) ...[
                          const SizedBox(height: 20),

                          Text(
                            'Selected image: '
                            '$_selectedImageName',

                            textAlign:
                                TextAlign.center,

                            style: const TextStyle(
                              color: Colors.grey,
                            ),
                          ),
                        ],

                        // =================================================
                        // SELECTED VIDEO
                        // =================================================

                        if (_selectedVideoName != null) ...[
                          const SizedBox(height: 20),

                          Text(
                            'Selected video: '
                            '$_selectedVideoName',

                            textAlign:
                                TextAlign.center,

                            style: const TextStyle(
                              color: Colors.grey,
                            ),
                          ),
                        ],

                        const SizedBox(height: 25),

                        // =================================================
                        // RESULT
                        // =================================================

                        _buildResultCard(),
                      ],
                    ),
                  ),

                  const SizedBox(height: 30),

                  // =================================================
                  // INFORMATION CARDS
                  // =================================================

                  Row(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,

                    children: [
                      Expanded(
                        child: _InfoCard(
                          icon: Icons.analytics,
                          title: 'Detection Status',

                          value:
                              _prediction == null
                                  ? 'Ready'
                                  : 'Completed',

                          description:
                              'The detection system is ready '
                              'for analysis.',
                        ),
                      ),

                      const SizedBox(width: 20),

                      Expanded(
                        child: _InfoCard(
                          icon: Icons.smart_toy,
                          title: 'AI Model',
                          value: 'CanineCue Model',

                          description:
                              'Computer vision model for '
                              'dog behavior analysis.',
                        ),
                      ),

                      const SizedBox(width: 20),

                      Expanded(
                        child: _InfoCard(
                          icon: Icons.shield,
                          title: 'Purpose',
                          value: 'Safety',

                          description:
                              'Helps identify potentially '
                              'aggressive behavior.',
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 40),

                  // =================================================
                  // ABOUT
                  // =================================================

                  Container(
                    width: double.infinity,

                    padding: const EdgeInsets.all(25),

                    decoration: BoxDecoration(
                      color: Colors.indigo,
                      borderRadius:
                          BorderRadius.circular(18),
                    ),

                    child: const Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start,

                      children: [
                        Text(
                          'About CanineCue',

                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 23,
                            fontWeight: FontWeight.bold,
                          ),
                        ),

                        SizedBox(height: 10),

                        Text(
                          'CanineCue is a dog aggression detection '
                          'system designed to analyze visual behavior '
                          'and identify potentially aggressive actions '
                          'using artificial intelligence and '
                          'computer vision.',

                          style: TextStyle(
                            color: Colors.white70,
                            fontSize: 15,
                            height: 1.5,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 30),

                  const Center(
                    child: Text(
                      'CanineCue • Dog Aggression Detection System',

                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 13,
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================
// RESULT ROW
// =============================================================

class _ResultRow extends StatelessWidget {
  final String label;
  final String value;

  const _ResultRow({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding:
          const EdgeInsets.symmetric(vertical: 5),

      child: Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,

        children: [
          SizedBox(
            width: 160,

            child: Text(
              label,

              style: const TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),

          Expanded(
            child: Text(value),
          ),
        ],
      ),
    );
  }
}

// =============================================================
// INFORMATION CARD
// =============================================================

class _InfoCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;
  final String description;

  const _InfoCard({
    required this.icon,
    required this.title,
    required this.value,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(22),

      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius:
            BorderRadius.circular(16),

        boxShadow: [
          BoxShadow(
            blurRadius: 12,
            color:
                Colors.black.withOpacity(0.05),
          ),
        ],
      ),

      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,

        children: [
          Icon(
            icon,
            color: Colors.indigo,
            size: 32,
          ),

          const SizedBox(height: 15),

          Text(
            title,

            style: const TextStyle(
              color: Colors.grey,
              fontSize: 14,
            ),
          ),

          const SizedBox(height: 5),

          Text(
            value,

            style: const TextStyle(
              fontSize: 19,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 8),

          Text(
            description,

            style: const TextStyle(
              color: Colors.grey,
              fontSize: 13,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}