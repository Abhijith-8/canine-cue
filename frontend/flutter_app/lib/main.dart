import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CanineCueApp());
}

// ============================================================
// ON-DEVICE DOG DETECTOR
// ============================================================

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

      final confidence = isAggressive
          ? probability
          : 1.0 - probability;

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

// ============================================================
// APP
// ============================================================

class CanineCueApp extends StatelessWidget {
  const CanineCueApp({super.key});

  @override
  Widget build(BuildContext context) {
    const amber = Color(0xFFFFA726);
    const background = Color(0xFF111111);

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'CanineCue',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: background,
        colorScheme: ColorScheme.fromSeed(
          seedColor: amber,
          brightness: Brightness.dark,
        ),
        fontFamily: 'Arial',
      ),
      home: const HomePage(),
    );
  }
}

// ============================================================
// HOME PAGE
// ============================================================

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const Color amber = Color(0xFFFFA726);
  static const Color background = Color(0xFF111111);
  static const Color card = Color(0xFF1B1B1B);
  static const Color card2 = Color(0xFF222222);

  final ImagePicker _picker = ImagePicker();
  final OnDeviceDogDetector _detector = OnDeviceDogDetector();

  static const MethodChannel _videoChannel =
      MethodChannel('caninecue/video');

  bool _isLoading = false;

  String? _selectedImageName;
  String? _selectedVideoName;

  String? _prediction;
  double? _confidence;

  int? _aggressiveFrames;
  int? _calmFrames;
  int? _analyzedFrames;

  double? _aggressivePercentage;

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

  void _resetResult() {
    _prediction = null;
    _confidence = null;
    _aggressiveFrames = null;
    _calmFrames = null;
    _analyzedFrames = null;
    _aggressivePercentage = null;
  }

  // ==========================================================
  // CAMERA OPTIONS
  // ==========================================================

  Future<void> _showCameraOptions() async {
    if (_isLoading) return;

    await showModalBottomSheet(
      context: context,
      backgroundColor: card,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(24),
        ),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 42,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.25),
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                const SizedBox(height: 20),
                const Text(
                  'Camera Detection',
                  style: TextStyle(
                    fontSize: 21,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 20),
                _BottomSheetButton(
                  icon: Icons.camera_alt_rounded,
                  title: 'Take Photo',
                  onTap: () {
                    Navigator.pop(context);
                    _takePhoto();
                  },
                ),
                const SizedBox(height: 12),
                _BottomSheetButton(
                  icon: Icons.videocam_rounded,
                  title: 'Record Video',
                  onTap: () {
                    Navigator.pop(context);
                    _recordVideo();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ==========================================================
  // PHOTO CAMERA
  // ==========================================================

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

  // ==========================================================
  // VIDEO ANALYSIS
  // ==========================================================

  Future<Map<String, dynamic>> _analyzeVideoOnDevice(
    String videoPath,
  ) async {
    final frames = await _videoChannel.invokeMethod<List<dynamic>>(
      'extractFrames',
      {
        'videoPath': videoPath,
        'frameCount': 30,
      },
    );

    if (frames == null || frames.isEmpty) {
      throw Exception(
        'No frames were extracted from the video.',
      );
    }

    int aggressiveFrames = 0;
    int calmFrames = 0;

    double confidenceSum = 0.0;
    double probabilitySum = 0.0;

    for (int i = 0; i < frames.length; i++) {
      final dynamic frame = frames[i];

      final Uint8List bytes = frame is Uint8List
          ? frame
          : Uint8List.fromList(
              List<int>.from(frame as List),
            );

      final data = await _detector.predict(bytes);

      final prediction = data['prediction'] as String;
      final confidence =
          (data['confidence'] as num).toDouble();
      final probability =
          (data['prob_aggressive'] as num).toDouble();

      if (prediction == 'AGGRESSIVE') {
        aggressiveFrames++;
      } else {
        calmFrames++;
      }

      confidenceSum += confidence;
      probabilitySum += probability;

      if (mounted) {
        setState(() {
          _analyzedFrames = i + 1;
        });
      }
    }

    final analyzedFrames = aggressiveFrames + calmFrames;

    final aggressivePercentage =
        (aggressiveFrames / analyzedFrames) * 100.0;

    final finalPrediction =
        aggressiveFrames > calmFrames
            ? 'AGGRESSIVE'
            : 'CALM';

    final averageConfidence =
        confidenceSum / analyzedFrames;

    final averageProbability =
        probabilitySum / analyzedFrames;

    return {
      'prediction': finalPrediction,
      'confidence': averageConfidence,
      'prob_aggressive': averageProbability,
      'aggressive_frames': aggressiveFrames,
      'calm_frames': calmFrames,
      'analyzed_frames': analyzedFrames,
      'aggressive_percentage': aggressivePercentage,
    };
  }

  // ==========================================================
  // RECORD VIDEO
  // ==========================================================

  Future<void> _recordVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.camera,
      );

      if (video == null) return;

      setState(() {
        _selectedVideoName = video.name;
        _selectedImageName = null;
        _resetResult();
        _isLoading = true;
      });

      final data = await _analyzeVideoOnDevice(
        video.path,
      );

      if (!mounted) return;

      setState(() {
        _prediction = data['prediction'] as String;
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
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _isLoading = false;
      });

      _showError('Video camera error: $e');
    }
  }

  // ==========================================================
  // GALLERY IMAGE
  // ==========================================================

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
        _confidence =
            (data['confidence'] as num).toDouble();
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

  // ==========================================================
  // GALLERY VIDEO
  // ==========================================================

  Future<void> _pickVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.gallery,
      );

      if (video == null) return;

      setState(() {
        _selectedVideoName = video.name;
        _selectedImageName = null;
        _resetResult();
        _isLoading = true;
      });

      final data = await _analyzeVideoOnDevice(
        video.path,
      );

      if (!mounted) return;

      setState(() {
        _prediction = data['prediction'] as String;
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
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _isLoading = false;
      });

      _showError('Video detection error: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 5),
      ),
    );
  }

  // ==========================================================
  // RESULT CARD
  // ==========================================================

  Widget _buildResultCard() {
    if (_isLoading) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(28),
        decoration: BoxDecoration(
          color: card,
          borderRadius: BorderRadius.circular(22),
        ),
        child: const Column(
          children: [
            CircularProgressIndicator(
              color: amber,
            ),
            SizedBox(height: 18),
            Text(
              'Analyzing...',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'CanineCue is analyzing the selected input on your device.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white60,
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

    final Color resultColor =
        aggressive
            ? const Color(0xFFFF5252)
            : const Color(0xFF66BB6A);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: card,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: resultColor.withValues(alpha: 0.35),
        ),
      ),
      child: Column(
        children: [
          Container(
            width: 66,
            height: 66,
            decoration: BoxDecoration(
              color: resultColor.withValues(alpha: 0.12),
              shape: BoxShape.circle,
            ),
            child: Icon(
              aggressive
                  ? Icons.warning_rounded
                  : Icons.check_circle_rounded,
              color: resultColor,
              size: 38,
            ),
          ),
          const SizedBox(height: 14),
          Text(
            aggressive ? 'HIGHER RISK' : 'LOWER RISK',
            style: TextStyle(
              color: resultColor,
              fontSize: 24,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            _prediction!,
            style: const TextStyle(
              fontSize: 16,
              color: Colors.white70,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 14),
          Text(
            '${((_confidence ?? 0) * 100).toStringAsFixed(1)}%',
            style: const TextStyle(
              fontSize: 38,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Text(
            'confidence',
            style: TextStyle(
              color: Colors.white54,
            ),
          ),
          if (_selectedVideoName != null) ...[
            const SizedBox(height: 22),
            Divider(
              color: Colors.white.withValues(alpha: 0.10),
            ),
            const SizedBox(height: 10),
            _DarkInfoRow(
              label: 'Video',
              value: _selectedVideoName!,
            ),
            _DarkInfoRow(
              label: 'Frames analyzed',
              value: '${_analyzedFrames ?? 0}',
            ),
            _DarkInfoRow(
              label: 'Aggressive frames',
              value: '${_aggressiveFrames ?? 0}',
            ),
            _DarkInfoRow(
              label: 'Calm frames',
              value: '${_calmFrames ?? 0}',
            ),
            _DarkInfoRow(
              label: 'Aggressive percentage',
              value:
                  '${(_aggressivePercentage ?? 0).toStringAsFixed(1)}%',
            ),
          ],
        ],
      ),
    );
  }

  // ==========================================================
  // MAIN UI
  // ==========================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 18, 20, 30),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(
                maxWidth: 900,
              ),
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  _buildHeader(),
                  const SizedBox(height: 34),
                  _buildHero(),
                  const SizedBox(height: 24),
                  _buildScanCard(),
                  const SizedBox(height: 24),
                  _buildResultCard(),
                  const SizedBox(height: 30),
                  _buildHowItWorks(),
                  const SizedBox(height: 24),
                  _buildSafetyNote(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ==========================================================
  // HEADER
  // ==========================================================

  Widget _buildHeader() {
    return Row(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: amber.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(15),
          ),
          child: const Icon(
            Icons.pets_rounded,
            color: amber,
            size: 28,
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Text(
            'CanineCue',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(
            horizontal: 11,
            vertical: 7,
          ),
          decoration: BoxDecoration(
            color: const Color(0xFF17351F),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: const Color(0xFF2C6E3F),
            ),
          ),
          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.wifi_off_rounded,
                size: 14,
                color: Color(0xFF81C784),
              ),
              SizedBox(width: 5),
              Text(
                'OFFLINE',
                style: TextStyle(
                  color: Color(0xFF81C784),
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.6,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ==========================================================
  // HERO
  // ==========================================================

  Widget _buildHero() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Know the risk.\nStay safe.',
          style: TextStyle(
            fontSize: 40,
            height: 1.05,
            fontWeight: FontWeight.w800,
            letterSpacing: -1,
          ),
        ),
        const SizedBox(height: 12),
        Text(
          'Use on-device AI to analyze a dog and identify '
          'potentially aggressive behavior.',
          style: TextStyle(
            fontSize: 16,
            height: 1.5,
            color: Colors.white.withValues(alpha: 0.60),
          ),
        ),
      ],
    );
  }

  // ==========================================================
  // SCAN CARD
  // ==========================================================

  Widget _buildScanCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: card,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.07),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: amber.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: const Icon(
                  Icons.shield_rounded,
                  color: amber,
                ),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Dog Safety Scan',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 3),
                    Text(
                      'Single-dog analysis',
                      style: TextStyle(
                        color: Colors.white54,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 22),
          SizedBox(
            width: double.infinity,
            height: 58,
            child: ElevatedButton.icon(
              onPressed:
                  _isLoading ? null : _showCameraOptions,
              icon: const Icon(
                Icons.center_focus_strong_rounded,
              ),
              label: const Text(
                'SCAN A DOG',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: amber,
                foregroundColor: Colors.black,
                disabledBackgroundColor:
                    amber.withValues(alpha: 0.35),
                disabledForegroundColor:
                    Colors.black54,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _ActionButton(
                  icon: Icons.photo_rounded,
                  label: 'Photo',
                  onPressed:
                      _isLoading ? null : _pickImage,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _ActionButton(
                  icon: Icons.video_library_rounded,
                  label: 'Video',
                  onPressed:
                      _isLoading ? null : _pickVideo,
                ),
              ),
            ],
          ),
          if (_selectedImageName != null)
            _SelectedFile(
              icon: Icons.image_rounded,
              name: _selectedImageName!,
            ),
          if (_selectedVideoName != null)
            _SelectedFile(
              icon: Icons.video_file_rounded,
              name: _selectedVideoName!,
            ),
        ],
      ),
    );
  }

  // ==========================================================
  // HOW IT WORKS
  // ==========================================================

  Widget _buildHowItWorks() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'How CanineCue works',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 14),
        LayoutBuilder(
          builder: (context, constraints) {
            final compact = constraints.maxWidth < 650;

            final items = [
              const _FeatureCard(
                number: '01',
                icon: Icons.camera_alt_rounded,
                title: 'Capture',
                description:
                    'Point your phone at a single dog or choose a saved photo or video.',
              ),
              const _FeatureCard(
                number: '02',
                icon: Icons.memory_rounded,
                title: 'Analyze',
                description:
                    'The trained CanineCue model analyzes the input directly on the device.',
              ),
              const _FeatureCard(
                number: '03',
                icon: Icons.shield_rounded,
                title: 'Understand',
                description:
                    'Get a clear calm or aggressive result with confidence information.',
              ),
            ];

            if (compact) {
              return Column(
                children: [
                  items[0],
                  const SizedBox(height: 10),
                  items[1],
                  const SizedBox(height: 10),
                  items[2],
                ],
              );
            }

            return Row(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Expanded(child: items[0]),
                const SizedBox(width: 10),
                Expanded(child: items[1]),
                const SizedBox(width: 10),
                Expanded(child: items[2]),
              ],
            );
          },
        ),
      ],
    );
  }

  // ==========================================================
  // SAFETY NOTE
  // ==========================================================

  Widget _buildSafetyNote() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: amber.withValues(alpha: 0.07),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: amber.withValues(alpha: 0.18),
        ),
      ),
      child: const Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.info_outline_rounded,
            color: amber,
            size: 22,
          ),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'CanineCue is a safety-support tool. '
              'Keep a safe distance from unfamiliar dogs '
              'and do not approach a dog based only on the app result.',
              style: TextStyle(
                color: Colors.white70,
                height: 1.45,
                fontSize: 13,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// ACTION BUTTON
// ============================================================

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: onPressed,
      icon: Icon(
        icon,
        size: 19,
      ),
      label: Text(
        label,
        style: const TextStyle(
          fontWeight: FontWeight.w700,
        ),
      ),
      style: OutlinedButton.styleFrom(
        foregroundColor: Colors.white,
        side: BorderSide(
          color: Colors.white.withValues(alpha: 0.12),
        ),
        backgroundColor:
            Colors.white.withValues(alpha: 0.03),
        minimumSize: const Size(0, 48),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(14),
        ),
      ),
    );
  }
}

// ============================================================
// SELECTED FILE
// ============================================================

class _SelectedFile extends StatelessWidget {
  final IconData icon;
  final String name;

  const _SelectedFile({
    required this.icon,
    required this.name,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 14),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const Icon(
            Icons.check_circle_rounded,
            color: Color(0xFF81C784),
            size: 18,
          ),
          const SizedBox(width: 8),
          Icon(
            icon,
            size: 18,
            color: Colors.white54,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              name,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 13,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// FEATURE CARD
// ============================================================

class _FeatureCard extends StatelessWidget {
  final String number;
  final IconData icon;
  final String title;
  final String description;

  const _FeatureCard({
    required this.number,
    required this.icon,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1B1B),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.06),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                number,
                style: const TextStyle(
                  color: Color(0xFFFFA726),
                  fontWeight: FontWeight.w800,
                  fontSize: 12,
                ),
              ),
              const Spacer(),
              Icon(
                icon,
                color: const Color(0xFFFFA726),
                size: 22,
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            title,
            style: const TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 7),
          Text(
            description,
            style: const TextStyle(
              color: Colors.white54,
              height: 1.4,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// DARK RESULT ROW
// ============================================================

class _DarkInfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _DarkInfoRow({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 150,
            child: Text(
              label,
              style: const TextStyle(
                color: Colors.white54,
                fontSize: 13,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// BOTTOM SHEET BUTTON
// ============================================================

class _BottomSheetButton extends StatelessWidget {
  final IconData icon;
  final String title;
  final VoidCallback onTap;

  const _BottomSheetButton({
    required this.icon,
    required this.title,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 55,
      child: ElevatedButton.icon(
        onPressed: onTap,
        icon: Icon(icon),
        label: Text(
          title,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFFFFA726),
          foregroundColor: Colors.black,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
    );
  }
}
