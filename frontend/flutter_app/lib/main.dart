import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CanineCueApp());
}

class OnDeviceDogDetector {
  final OnnxRuntime _ort = OnnxRuntime();
  OrtSession? _session;

  Future<void> initialize() async {
    if (_session != null) return;
    _session = await _ort.createSessionFromAsset(
      'assets/caninecue_resnet18.onnx',
    );
  }

  Future<Map<String, dynamic>> predict(Uint8List imageBytes) async {
    await initialize();

    final image = img.decodeImage(imageBytes);
    if (image == null) {
      throw Exception('Could not decode image.');
    }

    final resized = img.copyResize(
      image,
      width: 224,
      height: 224,
    );

    final input = Float32List(1 * 3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    int index = 0;

    for (int channel = 0; channel < 3; channel++) {
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          double value;

          if (channel == 0) {
            value = pixel.r.toDouble();
          } else if (channel == 1) {
            value = pixel.g.toDouble();
          } else {
            value = pixel.b.toDouble();
          }

          input[index++] =
              (value / 255.0 - mean[channel]) / std[channel];
        }
      }
    }

    final inputValue = await OrtValue.fromList(
      input,
      [1, 3, 224, 224],
    );

    try {
      final session = _session!;
      final outputs = await session.run({
        session.inputNames.first: inputValue,
      });

      final outputValue = outputs[session.outputNames.first];
      if (outputValue == null) {
        throw Exception('ONNX model returned no output.');
      }

      final output = await outputValue.asList();
      final logit = _extractLogit(output);
      final probability = 1.0 / (1.0 + exp(-logit));
      final isAggressive = probability >= 0.5;
      final confidence = isAggressive ? probability : 1.0 - probability;

      outputValue.dispose();

      return {
        'prediction': isAggressive ? 'AGGRESSIVE' : 'CALM',
        'confidence': confidence,
        'prob_aggressive': probability,
      };
    } finally {
      inputValue.dispose();
    }
  }

  double _extractLogit(dynamic output) {
    dynamic value = output;
    while (value is List && value.isNotEmpty) {
      value = value[0];
    }
    if (value is num) {
      return value.toDouble();
    }
    throw Exception('Unexpected ONNX output format.');
  }

  Future<void> dispose() async {
    await _session?.close();
    _session = null;
  }
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
  final OnDeviceDogDetector _detector = OnDeviceDogDetector();

  // Video detection still uses the existing backend until local video
  // frame extraction is added. Photo detection is fully on-device.
  static const String backendUrl = 'http://127.0.0.1:8000';

  @override
  void initState() {
    super.initState();
    _detector.initialize().catchError((error) {
      debugPrint('ONNX initialization error: $error');
    });
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

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
  // RESET RESULT
  // =========================================================

  void _resetResult() {
    _prediction = null;
    _confidence = null;

    _aggressiveFrames = null;
    _calmFrames = null;
    _analyzedFrames = null;

    _aggressivePercentage = null;
    _durationSeconds = null;
  }

  // =========================================================
  // CAMERA OPTIONS
  // =========================================================

  Future<void> _showCameraOptions() async {
    if (_isLoading) {
      return;
    }

    await showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(20),
        ),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Camera Detection',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),

                const SizedBox(height: 20),

                // TAKE PHOTO
                SizedBox(
                  width: double.infinity,
                  height: 55,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Navigator.pop(context);
                      _takePhoto();
                    },
                    icon: const Icon(Icons.camera_alt),
                    label: const Text(
                      'Take Photo',
                      style: TextStyle(
                        fontSize: 16,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 12),

                // RECORD VIDEO
                SizedBox(
                  width: double.infinity,
                  height: 55,
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.pop(context);
                      _recordVideo();
                    },
                    icon: const Icon(Icons.videocam),
                    label: const Text(
                      'Record Video',
                      style: TextStyle(
                        fontSize: 16,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 10),
              ],
            ),
          ),
        );
      },
    );
  }

  // =========================================================
  // TAKE PHOTO USING PHONE CAMERA
  // =========================================================

  Future<void> _takePhoto() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
      );

      if (image == null) return;

      setState(() {
        _selectedImageName = image.name;
        _selectedVideoName = null;
        _resetResult();
        _isLoading = true;
      });

      final bytes = await image.readAsBytes();
      final data = await _detector.predict(bytes);

      if (!mounted) return;

      setState(() {
        _prediction = data['prediction'] as String;
        _confidence = (data['confidence'] as num).toDouble();
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
      _showError('Camera detection error: $e');
    }
  }

  // =========================================================
  // RECORD VIDEO USING PHONE CAMERA
  // =========================================================

  Future<void> _recordVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.camera,
      );

      if (video == null) {
        return;
      }

      setState(() {
        _selectedVideoName = video.name;
        _selectedImageName = null;

        _resetResult();

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

      final data = json.decode(responseBody);

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

      _showError(
        'Video camera error: $e',
      );
    }
  }

  // =========================================================
  // UPLOAD IMAGE FROM GALLERY
  // =========================================================

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
      );

      if (image == null) return;

      setState(() {
        _selectedImageName = image.name;
        _selectedVideoName = null;
        _resetResult();
        _isLoading = true;
      });

      final bytes = await image.readAsBytes();
      final data = await _detector.predict(bytes);

      if (!mounted) return;

      setState(() {
        _prediction = data['prediction'] as String;
        _confidence = (data['confidence'] as num).toDouble();
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
      _showError('Image detection error: $e');
    }
  }

  // =========================================================
  // UPLOAD VIDEO FROM GALLERY
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

        _resetResult();

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

      final data = json.decode(responseBody);

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
        duration: const Duration(seconds: 5),
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
        _prediction!.toLowerCase() == 'aggressive';

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
            aggressive
                ? 'AGGRESSIVE'
                : 'CALM',
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

          // VIDEO INFORMATION
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
      backgroundColor:
          const Color(0xFFF5F7FB),

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
          if (MediaQuery.of(context).size.width > 700)
            TextButton(
              onPressed: () {},
              child: const Text('Home'),
            ),

          if (MediaQuery.of(context).size.width > 700)
            TextButton(
              onPressed: () {},
              child: const Text('History'),
            ),

          if (MediaQuery.of(context).size.width > 700)
            TextButton(
              onPressed: () {},
              child: const Text('About'),
            ),

          if (MediaQuery.of(context).size.width > 700)
            const SizedBox(width: 15),
        ],
      ),

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
                          color: Colors.black
                              .withValues(alpha: 0.06),
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
                            fontWeight:
                                FontWeight.bold,
                          ),
                        ),

                        const SizedBox(height: 10),

                        const Text(
                          'Choose an option below to analyze '
                          'dog behavior.',
                          textAlign:
                              TextAlign.center,
                          style: TextStyle(
                            fontSize: 15,
                            color: Colors.grey,
                          ),
                        ),

                        const SizedBox(height: 30),

                        // =================================================
                        // CAMERA DETECTION
                        // =================================================

                        SizedBox(
                          width: 260,
                          height: 55,

                          child:
                              ElevatedButton.icon(
                            onPressed:
                                _isLoading
                                    ? null
                                    : _showCameraOptions,

                            icon: const Icon(
                              Icons.camera_alt,
                            ),

                            label: const Text(
                              'Camera Detection',
                              style: TextStyle(
                                fontSize: 16,
                              ),
                            ),

                            style:
                                ElevatedButton
                                    .styleFrom(
                              backgroundColor:
                                  Colors.indigo,

                              foregroundColor:
                                  Colors.white,

                              shape:
                                  RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius
                                        .circular(12),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 15),

                        // =================================================
                        // UPLOAD IMAGE
                        // =================================================

                        SizedBox(
                          width: 260,
                          height: 55,

                          child:
                              OutlinedButton.icon(
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
                                OutlinedButton
                                    .styleFrom(
                              foregroundColor:
                                  Colors.indigo,

                              side:
                                  const BorderSide(
                                color: Colors.indigo,
                              ),

                              shape:
                                  RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius
                                        .circular(12),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 15),

                        // =================================================
                        // UPLOAD VIDEO
                        // =================================================

                        SizedBox(
                          width: 260,
                          height: 55,

                          child:
                              OutlinedButton.icon(
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
                                OutlinedButton
                                    .styleFrom(
                              foregroundColor:
                                  Colors.indigo,

                              side:
                                  const BorderSide(
                                color: Colors.indigo,
                              ),

                              shape:
                                  RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius
                                        .circular(12),
                              ),
                            ),
                          ),
                        ),

                        // =================================================
                        // SELECTED IMAGE
                        // =================================================

                        if (_selectedImageName !=
                            null) ...[
                          const SizedBox(height: 20),

                          Text(
                            'Selected image: '
                            '$_selectedImageName',

                            textAlign:
                                TextAlign.center,

                            style:
                                const TextStyle(
                              color: Colors.grey,
                            ),
                          ),
                        ],

                        // =================================================
                        // SELECTED VIDEO
                        // =================================================

                        if (_selectedVideoName !=
                            null) ...[
                          const SizedBox(height: 20),

                          Text(
                            'Selected video: '
                            '$_selectedVideoName',

                            textAlign:
                                TextAlign.center,

                            style:
                                const TextStyle(
                              color: Colors.grey,
                            ),
                          ),
                        ],

                        const SizedBox(height: 25),

                        // RESULT
                        _buildResultCard(),
                      ],
                    ),
                  ),

                  const SizedBox(height: 30),

                  // =================================================
                  // INFORMATION CARDS
                  // =================================================

                  LayoutBuilder(
                    builder:
                        (context, constraints) {
                      if (constraints.maxWidth <
                          700) {
                        return Column(
                          children: [
                            _InfoCard(
                              icon:
                                  Icons.analytics,
                              title:
                                  'Detection Status',
                              value:
                                  _prediction ==
                                          null
                                      ? 'Ready'
                                      : 'Completed',
                              description:
                                  'The detection system is ready '
                                  'for analysis.',
                            ),

                            const SizedBox(
                              height: 20,
                            ),

                            _InfoCard(
                              icon:
                                  Icons.smart_toy,
                              title: 'AI Model',
                              value:
                                  'CanineCue Model',
                              description:
                                  'Computer vision model for '
                                  'dog behavior analysis.',
                            ),

                            const SizedBox(
                              height: 20,
                            ),

                            _InfoCard(
                              icon: Icons.shield,
                              title: 'Purpose',
                              value: 'Safety',
                              description:
                                  'Helps identify potentially '
                                  'aggressive behavior.',
                            ),
                          ],
                        );
                      }

                      return Row(
                        crossAxisAlignment:
                            CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            child: _InfoCard(
                              icon:
                                  Icons.analytics,
                              title:
                                  'Detection Status',
                              value:
                                  _prediction ==
                                          null
                                      ? 'Ready'
                                      : 'Completed',
                              description:
                                  'The detection system is ready '
                                  'for analysis.',
                            ),
                          ),

                          const SizedBox(
                            width: 20,
                          ),

                          Expanded(
                            child: _InfoCard(
                              icon:
                                  Icons.smart_toy,
                              title: 'AI Model',
                              value:
                                  'CanineCue Model',
                              description:
                                  'Computer vision model for '
                                  'dog behavior analysis.',
                            ),
                          ),

                          const SizedBox(
                            width: 20,
                          ),

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
                      );
                    },
                  ),

                  const SizedBox(height: 40),

                  // =================================================
                  // ABOUT
                  // =================================================

                  Container(
                    width: double.infinity,

                    padding:
                        const EdgeInsets.all(25),

                    decoration: BoxDecoration(
                      color: Colors.white,

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
                            fontSize: 24,
                            fontWeight:
                                FontWeight.bold,
                          ),
                        ),

                        SizedBox(height: 10),

                        Text(
                          'CanineCue uses AI-powered computer '
                          'vision to analyze dog behavior and '
                          'identify potentially aggressive '
                          'activity.',
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// =========================================================
// RESULT ROW
// =========================================================

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
            width: 150,

            child: Text(
              label,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),

          Expanded(
            child: Text(
              value,
            ),
          ),
        ],
      ),
    );
  }
}

// =========================================================
// INFORMATION CARD
// =========================================================

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
      width: double.infinity,

      padding: const EdgeInsets.all(25),

      decoration: BoxDecoration(
        color: Colors.white,

        borderRadius:
            BorderRadius.circular(18),

        boxShadow: [
          BoxShadow(
            blurRadius: 12,
            spreadRadius: 1,
            color: Colors.black
                .withValues(alpha: 0.05),
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
            size: 35,
          ),

          const SizedBox(height: 15),

          Text(
            title,
            style: const TextStyle(
              color: Colors.grey,
              fontSize: 15,
            ),
          ),

          const SizedBox(height: 5),

          Text(
            value,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 8),

          Text(
            description,
            style: const TextStyle(
              color: Colors.grey,
            ),
          ),
        ],
      ),
    );
  }
}
