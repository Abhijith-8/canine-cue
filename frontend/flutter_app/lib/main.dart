import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:camera/camera.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(CanineCueApp(cameras: cameras));
}

// ============================================================
// CONSTANTS
// ============================================================

const Color kAmber      = Color(0xFFFFA726);
const Color kBackground = Color(0xFF111111);
const Color kCard       = Color(0xFF1B1B1B);
const Color kCard2      = Color(0xFF222222);
const Color kRed        = Color(0xFFFF5252);
const Color kGreen      = Color(0xFF66BB6A);
const Color kOrange     = Color(0xFFFFB74D);

// ============================================================
// API SERVICE
// ============================================================

class ApiService {
  static String baseUrl = 'https://your-ngrok-url.ngrok-free.app';

  static Future<void> loadSavedUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('ngrok_url');
    if (saved != null && saved.isNotEmpty) baseUrl = saved;
  }

  static Future<void> saveUrl(String url) async {
    baseUrl = url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('ngrok_url', url);
  }

  static Future<Map<String, dynamic>> predictImage(
      Uint8List imageBytes) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
      'image', imageBytes,
      filename: 'image.jpg',
    ));
    request.headers['ngrok-skip-browser-warning'] = 'true';

    final streamed = await request.send()
        .timeout(const Duration(seconds: 20));
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw Exception('Server error ${response.statusCode}');
  }

  static Future<Map<String, dynamic>> predictVideo(
      Uint8List videoBytes) async {
    final uri = Uri.parse('$baseUrl/predict/video');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
      'video', videoBytes,
      filename: 'video.mp4',
    ));
    request.headers['ngrok-skip-browser-warning'] = 'true';

    final streamed = await request.send()
        .timeout(const Duration(seconds: 120));
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw Exception('Server error ${response.statusCode}');
  }

  static Future<bool> checkHealth() async {
    try {
      final uri = Uri.parse('$baseUrl/health');
      final response = await http.get(uri, headers: {
        'ngrok-skip-browser-warning': 'true',
      }).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}

// ============================================================
// ON-DEVICE DETECTOR (OFFLINE)
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
    if (image == null) throw Exception('Could not decode image.');

    final resized = img.copyResize(image, width: 224, height: 224);
    final input = Float32List(1 * 3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std  = [0.229, 0.224, 0.225];
    int index = 0;

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          final value = c == 0
              ? pixel.r.toDouble()
              : c == 1
                  ? pixel.g.toDouble()
                  : pixel.b.toDouble();
          input[index++] = (value / 255.0 - mean[c]) / std[c];
        }
      }
    }

    final inputValue = await OrtValue.fromList(input, [1, 3, 224, 224]);

    try {
      final session = _session!;
      final outputs = await session.run({
        session.inputNames.first: inputValue,
      });
      final outputValue = outputs[session.outputNames.first];
      if (outputValue == null) throw Exception('No output from model.');

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
    while (value is List && value.isNotEmpty) value = value[0];
    if (value is num) return value.toDouble();
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
  final List<CameraDescription> cameras;
  const CanineCueApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'CanineCue',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: kBackground,
        colorScheme: ColorScheme.fromSeed(
          seedColor: kAmber,
          brightness: Brightness.dark,
        ),
        fontFamily: 'Arial',
      ),
      home: HomePage(cameras: cameras),
    );
  }
}

// ============================================================
// HOME PAGE
// ============================================================

class HomePage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const HomePage({super.key, required this.cameras});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  final OnDeviceDogDetector _detector = OnDeviceDogDetector();
  static const MethodChannel _videoChannel = MethodChannel('caninecue/video');

  // mode
  bool _isOnline = false;
  bool _serverOnline = false;

  // loading
  bool _isLoading = false;
  String _loadingMessage = 'Analyzing...';

  // selected
  String? _selectedImageName;
  String? _selectedVideoName;
  Uint8List? _selectedImageBytes;

  // result — offline
  String? _prediction;
  double? _confidence;
  int? _aggressiveFrames;
  int? _calmFrames;
  int? _analyzedFrames;
  double? _aggressivePercentage;

  // result — online
  bool? _detected;
  String? _riskLevel;
  double? _onlineConfidence;
  double? _probAggressive;
  List<double>? _bbox;

  // online video
  String? _overallRisk;
  double? _overallConfidence;
  int? _analyzedFramesOnline;
  int? _detectedFrames;
  int? _highFrames;
  int? _mediumFrames;
  int? _lowFrames;

  @override
  void initState() {
    super.initState();
    ApiService.loadSavedUrl();
    _detector.initialize().catchError((e) {
      debugPrint('ONNX init error: $e');
    });
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  // ── mode toggle ──────────────────────────────────────────

  Future<void> _toggleMode(bool online) async {
    if (online) {
      setState(() => _isLoading = true);
      final ok = await ApiService.checkHealth();
      setState(() {
        _isOnline = online;
        _serverOnline = ok;
        _isLoading = false;
      });
      if (!ok) {
        _showError(
            'Cannot reach server. Check ngrok URL in Settings.');
      }
    } else {
      setState(() {
        _isOnline = false;
        _serverOnline = false;
      });
    }
    _resetResult();
  }

  void _resetResult() {
    setState(() {
      _prediction = null;
      _confidence = null;
      _aggressiveFrames = null;
      _calmFrames = null;
      _analyzedFrames = null;
      _aggressivePercentage = null;
      _detected = null;
      _riskLevel = null;
      _onlineConfidence = null;
      _probAggressive = null;
      _bbox = null;
      _overallRisk = null;
      _overallConfidence = null;
      _analyzedFramesOnline = null;
      _detectedFrames = null;
      _highFrames = null;
      _mediumFrames = null;
      _lowFrames = null;
      _selectedImageBytes = null;
      _selectedImageName = null;
      _selectedVideoName = null;
    });
  }

  // ── camera options bottom sheet ──────────────────────────

  Future<void> _showCameraOptions() async {
    if (_isLoading) return;
    await showModalBottomSheet(
      context: context,
      backgroundColor: kCard,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 42, height: 4,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              const SizedBox(height: 20),
              const Text('Camera Detection',
                  style: TextStyle(
                      fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 20),
              _BottomSheetButton(
                icon: Icons.camera_alt_rounded,
                title: 'Take Photo',
                onTap: () { Navigator.pop(context); _takePhoto(); },
              ),
              const SizedBox(height: 10),
              if (_isOnline) ...[
                // _BottomSheetButton(
                //   icon: Icons.videocam_rounded,
                //   title: 'Record Video',
                //   onTap: () { Navigator.pop(context); _recordVideo(); },
                // ),
                // const SizedBox(height: 10),
                _BottomSheetButton(
                  icon: Icons.stream_rounded,
                  title: 'Live Camera Feed',
                  onTap: () {
                    Navigator.pop(context);
                    _openLiveCamera();
                  },
                ),
              ] else ...[
                _BottomSheetButton(
                  icon: Icons.videocam_rounded,
                  title: 'Record Video',
                  onTap: () { Navigator.pop(context); _recordVideo(); },
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  // ── take photo ───────────────────────────────────────────

  Future<void> _takePhoto() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera, imageQuality: 90);
      if (image == null) return;

      final bytes = await image.readAsBytes();
      setState(() {
        _selectedImageName = image.name;
        _selectedVideoName = null;
        _selectedImageBytes = bytes;
        _resetResult();
        _selectedImageName = image.name;
        _selectedImageBytes = bytes;
        _isLoading = true;
        _loadingMessage = 'Analyzing image...';
      });

      if (_isOnline) {
        final data = await ApiService.predictImage(bytes);
        if (!mounted) return;
        _applyOnlineImageResult(data);
      } else {
        final data = await _detector.predict(bytes);
        if (!mounted) return;
        _applyOfflineResult(data);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      _showError('Camera error: $e');
    }
  }

  // ── pick image ───────────────────────────────────────────

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery);
      if (image == null) return;

      final bytes = await image.readAsBytes();
      setState(() {
        _selectedImageName = image.name;
        _selectedVideoName = null;
        _selectedImageBytes = bytes;
        _resetResult();
        _selectedImageName = image.name;
        _selectedImageBytes = bytes;
        _isLoading = true;
        _loadingMessage = 'Analyzing image...';
      });

      if (_isOnline) {
        final data = await ApiService.predictImage(bytes);
        if (!mounted) return;
        _applyOnlineImageResult(data);
      } else {
        final data = await _detector.predict(bytes);
        if (!mounted) return;
        _applyOfflineResult(data);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      _showError('Image error: $e');
    }
  }

  // ── pick / record video ──────────────────────────────────

  Future<void> _pickVideo() async => _handleVideo(ImageSource.gallery);
  Future<void> _recordVideo() async => _handleVideo(ImageSource.camera);

  Future<void> _handleVideo(ImageSource source) async {
    try {
      final XFile? video = await _picker.pickVideo(source: source);
      if (video == null) return;

      setState(() {
        _selectedVideoName = video.name;
        _selectedImageName = null;
        _selectedImageBytes = null;
        _resetResult();
        _selectedVideoName = video.name;
        _isLoading = true;
        _loadingMessage = 'Processing video...';
      });

      if (_isOnline) {
        final bytes = await video.readAsBytes();
        final data = await ApiService.predictVideo(bytes);
        if (!mounted) return;
        _applyOnlineVideoResult(data);
      } else {
        final data = await _analyzeVideoOffline(video.path);
        if (!mounted) return;
        _applyOfflineVideoResult(data);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      _showError('Video error: $e');
    }
  }

  // ── offline video ────────────────────────────────────────

  Future<Map<String, dynamic>> _analyzeVideoOffline(
      String videoPath) async {
    final frames = await _videoChannel.invokeMethod<List<dynamic>>(
      'extractFrames',
      {'videoPath': videoPath, 'frameCount': 30},
    );
    if (frames == null || frames.isEmpty) {
      throw Exception('No frames extracted from video.');
    }

    int aggFrames = 0, calmFrames = 0;
    double confSum = 0, probSum = 0;

    for (int i = 0; i < frames.length; i++) {
      final bytes = frames[i] is Uint8List
          ? frames[i] as Uint8List
          : Uint8List.fromList(List<int>.from(frames[i] as List));

      final data = await _detector.predict(bytes);
      if (data['prediction'] == 'AGGRESSIVE') aggFrames++; else calmFrames++;
      confSum += (data['confidence'] as num).toDouble();
      probSum += (data['prob_aggressive'] as num).toDouble();

      if (mounted) setState(() => _analyzedFrames = i + 1);
    }

    final total = aggFrames + calmFrames;
    return {
      'prediction': aggFrames > calmFrames ? 'AGGRESSIVE' : 'CALM',
      'confidence': confSum / total,
      'prob_aggressive': probSum / total,
      'aggressive_frames': aggFrames,
      'calm_frames': calmFrames,
      'analyzed_frames': total,
      'aggressive_percentage': (aggFrames / total) * 100.0,
    };
  }

  // ── live camera ──────────────────────────────────────────

  void _openLiveCamera() {
    if (widget.cameras.isEmpty) {
      _showError('No camera available.');
      return;
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => LiveCameraScreen(
          camera: widget.cameras.first,
        ),
      ),
    );
  }

  // ── apply results ────────────────────────────────────────

  void _applyOfflineResult(Map<String, dynamic> data) {
    setState(() {
      _prediction = data['prediction'] as String;
      _confidence = (data['confidence'] as num).toDouble();
      _isLoading = false;
    });
  }

  void _applyOfflineVideoResult(Map<String, dynamic> data) {
    setState(() {
      _prediction = data['prediction'] as String;
      _confidence = (data['confidence'] as num).toDouble();
      _aggressiveFrames = (data['aggressive_frames'] as num).toInt();
      _calmFrames = (data['calm_frames'] as num).toInt();
      _analyzedFrames = (data['analyzed_frames'] as num).toInt();
      _aggressivePercentage =
          (data['aggressive_percentage'] as num).toDouble();
      _isLoading = false;
    });
  }

  void _applyOnlineImageResult(Map<String, dynamic> data) {
    setState(() {
      _detected = data['detected'] as bool? ?? false;
      _riskLevel = data['risk_level'] as String? ?? 'Unknown';
      _onlineConfidence = (data['confidence'] as num?)?.toDouble() ?? 0;
      _probAggressive =
          (data['prob_aggressive'] as num?)?.toDouble() ?? 0;
      if (data['bbox'] != null) {
        _bbox = (data['bbox'] as List)
            .map((v) => (v as num).toDouble())
            .toList();
      }
      _isLoading = false;
    });
  }

  void _applyOnlineVideoResult(Map<String, dynamic> data) {
    setState(() {
      _overallRisk = data['overall_risk'] as String? ?? 'Unknown';
      _overallConfidence =
          (data['overall_confidence'] as num?)?.toDouble() ?? 0;
      _analyzedFramesOnline =
          (data['analyzed_frames'] as num?)?.toInt() ?? 0;
      _detectedFrames = (data['detected_frames'] as num?)?.toInt() ?? 0;
      _highFrames = (data['high_risk_frames'] as num?)?.toInt() ?? 0;
      _mediumFrames = (data['medium_risk_frames'] as num?)?.toInt() ?? 0;
      _lowFrames = (data['low_risk_frames'] as num?)?.toInt() ?? 0;
      _isLoading = false;
    });
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: kRed.withValues(alpha: 0.9),
        duration: const Duration(seconds: 5),
      ),
    );
  }

  // ============================================================
  // BUILD
  // ============================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 18, 20, 40),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 900),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeader(),
                  const SizedBox(height: 28),
                  _buildModeToggle(),
                  const SizedBox(height: 28),
                  _buildHero(),
                  const SizedBox(height: 24),
                  _buildScanCard(),
                  const SizedBox(height: 20),
                  _buildResultSection(),
                  const SizedBox(height: 32),
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

  // ── header ───────────────────────────────────────────────

  Widget _buildHeader() {
    return Row(
      children: [
        Container(
          width: 46, height: 46,
          decoration: BoxDecoration(
            color: kAmber.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(14),
          ),
          child: const Icon(Icons.pets_rounded, color: kAmber, size: 26),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Text('CanineCue',
              style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800)),
        ),
        // Settings
        GestureDetector(
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const SettingsScreen()),
          ),
          child: Container(
            width: 40, height: 40,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.settings_rounded,
                color: Colors.white54, size: 20),
          ),
        ),
      ],
    );
  }

  // ── mode toggle ──────────────────────────────────────────

  Widget _buildModeToggle() {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.07)),
      ),
      child: Row(
        children: [
          _ModeTab(
            label: 'Offline',
            icon: Icons.wifi_off_rounded,
            active: !_isOnline,
            onTap: () => _toggleMode(false),
          ),
          _ModeTab(
            label: 'Online',
            icon: Icons.wifi_rounded,
            active: _isOnline,
            onTap: () => _toggleMode(true),
          ),
        ],
      ),
    );
  }

  // ── hero ─────────────────────────────────────────────────

  Widget _buildHero() {
    final subtitle = _isOnline
        ? 'Online mode — YOLO detection + ResNet18 classification via backend.'
        : 'Offline mode — on-device AI, no internet required.';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Know the risk.\nStay safe.',
          style: TextStyle(
            fontSize: 38,
            height: 1.08,
            fontWeight: FontWeight.w800,
            letterSpacing: -0.8,
          ),
        ),
        const SizedBox(height: 10),
        Text(
          subtitle,
          style: TextStyle(
            fontSize: 14,
            height: 1.5,
            color: Colors.white.withValues(alpha: 0.55),
          ),
        ),
        if (_isOnline) ...[
          const SizedBox(height: 10),
          _ServerStatusBadge(online: _serverOnline),
        ],
      ],
    );
  }

  // ── scan card ────────────────────────────────────────────

  Widget _buildScanCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Colors.white.withValues(alpha: 0.07)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 42, height: 42,
                decoration: BoxDecoration(
                  color: kAmber.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.shield_rounded,
                    color: kAmber, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Dog Safety Scan',
                        style: TextStyle(
                            fontSize: 17, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 2),
                    Text(
                      _isOnline
                          ? 'YOLO + ResNet18 via server'
                          : 'On-device ResNet18',
                      style: const TextStyle(
                          color: Colors.white54, fontSize: 12),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            height: 54,
            child: ElevatedButton.icon(
              onPressed: _isLoading ? null : _showCameraOptions,
              icon: const Icon(Icons.center_focus_strong_rounded),
              label: const Text('SCAN A DOG',
                  style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1)),
              style: ElevatedButton.styleFrom(
                backgroundColor: kAmber,
                foregroundColor: Colors.black,
                disabledBackgroundColor: kAmber.withValues(alpha: 0.3),
                disabledForegroundColor: Colors.black45,
                elevation: 0,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
            ),
          ),
          const SizedBox(height: 10),
          Row(
  children: [
    Expanded(
      child: _ActionButton(
        icon: Icons.photo_rounded,
        label: 'Photo',
        onPressed: _isLoading ? null : _pickImage,
      ),
    ),
    if (_isOnline) ...[
      const SizedBox(width: 10),
      Expanded(
        child: _ActionButton(
          icon: Icons.video_library_rounded,
          label: 'Video',
          onPressed: _isLoading ? null : _pickVideo,
        ),
      ),
      const SizedBox(width: 10),
      Expanded(
        child: _ActionButton(
          icon: Icons.stream_rounded,
          label: 'Live',
          onPressed: _isLoading ? null : _openLiveCamera,
        ),
      ),
    ],
  ],
),
          if (_selectedImageName != null)
            _SelectedFile(icon: Icons.image_rounded, name: _selectedImageName!),
          if (_selectedVideoName != null)
            _SelectedFile(icon: Icons.video_file_rounded, name: _selectedVideoName!),
        ],
      ),
    );
  }

  // ── result section ───────────────────────────────────────

  Widget _buildResultSection() {
    if (_isLoading) return _buildLoadingCard();
    if (_isOnline) return _buildOnlineResult();
    return _buildOfflineResult();
  }

  Widget _buildLoadingCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Column(
        children: [
          const CircularProgressIndicator(color: kAmber),
          const SizedBox(height: 16),
          Text(_loadingMessage,
              style: const TextStyle(
                  fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          Text(
            _isOnline
                ? 'Sending to server for analysis...'
                : 'Running on-device inference...',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white54, fontSize: 13),
          ),
          if (_analyzedFrames != null) ...[
            const SizedBox(height: 12),
            Text('Frames analyzed: $_analyzedFrames',
                style: const TextStyle(color: kAmber)),
          ],
        ],
      ),
    );
  }

  // offline result
  Widget _buildOfflineResult() {
    if (_prediction == null) return const SizedBox.shrink();

    final aggressive = _prediction!.toLowerCase() == 'aggressive';
    final color = aggressive ? kRed : kGreen;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          _RiskIcon(aggressive: aggressive, color: color),
          const SizedBox(height: 12),
          Text(
            aggressive ? 'HIGHER RISK' : 'LOWER RISK',
            style: TextStyle(
                color: color, fontSize: 22,
                fontWeight: FontWeight.w800, letterSpacing: 0.5),
          ),
          const SizedBox(height: 4),
          Text(_prediction!,
              style: const TextStyle(
                  fontSize: 14, color: Colors.white60,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 12),
          Text(
            '${((_confidence ?? 0) * 100).toStringAsFixed(1)}%',
            style: const TextStyle(
                fontSize: 36, fontWeight: FontWeight.bold),
          ),
          const Text('confidence',
              style: TextStyle(color: Colors.white54, fontSize: 13)),
          if (_selectedVideoName != null && _analyzedFrames != null) ...[
            const SizedBox(height: 18),
            Divider(color: Colors.white.withValues(alpha: 0.08)),
            const SizedBox(height: 8),
            _DarkInfoRow(label: 'Frames analyzed',
                value: '$_analyzedFrames'),
            _DarkInfoRow(label: 'Aggressive frames',
                value: '$_aggressiveFrames'),
            _DarkInfoRow(label: 'Calm frames', value: '$_calmFrames'),
            _DarkInfoRow(
                label: 'Aggressive %',
                value:
                    '${(_aggressivePercentage ?? 0).toStringAsFixed(1)}%'),
          ],
        ],
      ),
    );
  }

  // online result
  Widget _buildOnlineResult() {
    // video result
    if (_overallRisk != null) return _buildOnlineVideoResult();
    // image result
    if (_riskLevel == null) return const SizedBox.shrink();
    if (_detected == false) {
      return _buildNoDogCard();
    }
    return _buildOnlineImageResult();
  }

  Widget _buildNoDogCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
      ),
      child: const Column(
        children: [
          Icon(Icons.search_off_rounded, color: Colors.white38, size: 44),
          SizedBox(height: 12),
          Text('No dog detected',
              style: TextStyle(
                  fontSize: 18, fontWeight: FontWeight.bold)),
          SizedBox(height: 6),
          Text('Try a clearer photo with the dog more visible.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white54, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildOnlineImageResult() {
    final color = _riskColor(_riskLevel ?? 'Unknown');

    return Column(
      children: [
        // bbox image preview
        if (_selectedImageBytes != null && _bbox != null)
          _BboxImageWidget(
            imageBytes: _selectedImageBytes!,
            bbox: _bbox!,
            riskLevel: _riskLevel ?? 'Unknown',
          ),
        if (_selectedImageBytes != null && _bbox != null)
          const SizedBox(height: 16),

        // result card
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            color: kCard,
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: color.withValues(alpha: 0.3)),
          ),
          child: Column(
            children: [
              _RiskLevelBadge(riskLevel: _riskLevel ?? 'Unknown'),
              const SizedBox(height: 14),
              Text(
                '${((_onlineConfidence ?? 0) * 100).toStringAsFixed(1)}%',
                style: const TextStyle(
                    fontSize: 36, fontWeight: FontWeight.bold),
              ),
              const Text('confidence',
                  style: TextStyle(color: Colors.white54, fontSize: 13)),
              const SizedBox(height: 16),
              Divider(color: Colors.white.withValues(alpha: 0.08)),
              const SizedBox(height: 8),
              _DarkInfoRow(
                  label: 'Prob. aggressive',
                  value:
                      '${((_probAggressive ?? 0) * 100).toStringAsFixed(1)}%'),
              if (_bbox != null)
                _DarkInfoRow(
                    label: 'BBox',
                    value:
                        '[${_bbox!.map((v) => v.toStringAsFixed(0)).join(', ')}]'),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildOnlineVideoResult() {
    final color = _riskColor(_overallRisk ?? 'Unknown');
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          _RiskLevelBadge(riskLevel: _overallRisk ?? 'Unknown'),
          const SizedBox(height: 14),
          Text(
            '${((_overallConfidence ?? 0) * 100).toStringAsFixed(1)}%',
            style: const TextStyle(
                fontSize: 36, fontWeight: FontWeight.bold),
          ),
          const Text('overall confidence',
              style: TextStyle(color: Colors.white54, fontSize: 13)),
          const SizedBox(height: 16),
          Divider(color: Colors.white.withValues(alpha: 0.08)),
          const SizedBox(height: 8),
          _DarkInfoRow(
              label: 'Frames analyzed',
              value: '$_analyzedFramesOnline'),
          _DarkInfoRow(
              label: 'Dog detected in',
              value: '$_detectedFrames frames'),
          _DarkInfoRow(label: 'High risk frames', value: '$_highFrames'),
          _DarkInfoRow(
              label: 'Medium risk frames', value: '$_mediumFrames'),
          _DarkInfoRow(label: 'Low risk frames', value: '$_lowFrames'),
        ],
      ),
    );
  }

  // ── how it works ─────────────────────────────────────────

  Widget _buildHowItWorks() {
    final steps = _isOnline
        ? [
            ('01', Icons.cloud_upload_rounded, 'Upload',
                'Send a photo, video, or live stream to the CanineCue server.'),
            ('02', Icons.manage_search_rounded, 'Detect',
                'YOLOv8 locates the dog and crops it for precise analysis.'),
            ('03', Icons.shield_rounded, 'Classify',
                'ResNet18 classifies the cropped region and returns a risk level.'),
          ]
        : [
            ('01', Icons.camera_alt_rounded, 'Capture',
                'Point your phone at a dog or choose a saved photo or video.'),
            ('02', Icons.memory_rounded, 'Analyze',
                'On-device ResNet18 model runs entirely on your phone.'),
            ('03', Icons.shield_rounded, 'Understand',
                'Get a calm or aggressive result with confidence score.'),
          ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('How CanineCue works',
            style:
                TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
        const SizedBox(height: 14),
        LayoutBuilder(builder: (context, constraints) {
          final compact = constraints.maxWidth < 600;
          final cards = steps
              .map((s) => _FeatureCard(
                    number: s.$1,
                    icon: s.$2,
                    title: s.$3,
                    description: s.$4,
                  ))
              .toList();

          if (compact) {
            return Column(
              children: cards
                  .expand((c) => [c, const SizedBox(height: 10)])
                  .toList()
                ..removeLast(),
            );
          }
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: cards
                .expand((c) => [Expanded(child: c), const SizedBox(width: 10)])
                .toList()
              ..removeLast(),
          );
        }),
      ],
    );
  }

  // ── safety note ──────────────────────────────────────────

  Widget _buildSafetyNote() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kAmber.withValues(alpha: 0.07),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: kAmber.withValues(alpha: 0.18)),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline_rounded, color: kAmber, size: 20),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'CanineCue is a safety-support tool. '
              'Keep a safe distance from unfamiliar dogs '
              'and do not rely solely on the app result.',
              style: TextStyle(
                  color: Colors.white60, height: 1.45, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Color _riskColor(String level) {
    switch (level) {
      case 'High':   return kRed;
      case 'Medium': return kOrange;
      case 'Low':    return kGreen;
      default:       return Colors.grey;
    }
  }
}

// ============================================================
// LIVE CAMERA SCREEN
// ============================================================

// ============================================================
// LIVE CAMERA SCREEN
// ============================================================

class LiveCameraScreen extends StatefulWidget {
  final CameraDescription camera;
  const LiveCameraScreen({super.key, required this.camera});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  late CameraController _controller;
  late Future<void> _initFuture;

  bool _analyzing = false;
  Map<String, dynamic>? _lastResult;
  Size? _capturedImageSize; // actual size of image sent to server
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera, ResolutionPreset.medium, enableAudio: false);
    _initFuture = _controller.initialize();
    _startAnalysis();
  }

  void _startAnalysis() {
    _timer = Timer.periodic(const Duration(milliseconds: 800), (_) {
      _analyzeFrame();
    });
  }

  Future<void> _analyzeFrame() async {
    if (_analyzing || !_controller.value.isInitialized) return;
    _analyzing = true;

    try {
      final xFile = await _controller.takePicture();
      final bytes = await xFile.readAsBytes();

      // decode to get actual image dimensions
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final imageW = frame.image.width.toDouble();
      final imageH = frame.image.height.toDouble();

      final result = await ApiService.predictImage(bytes);

      if (mounted) {
        setState(() {
          _lastResult = result;
          _capturedImageSize = Size(imageW, imageH);
        });
      }
    } catch (_) {}

    _analyzing = false;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: FutureBuilder<void>(
        future: _initFuture,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(
                child: CircularProgressIndicator(color: kAmber));
          }
          return Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_controller),
              if (_lastResult != null && _capturedImageSize != null)
                _LiveOverlay(
                  result: _lastResult!,
                  capturedImageSize: _capturedImageSize!,
                ),
              // back button
              Positioned(
                top: 48, left: 16,
                child: GestureDetector(
                  onTap: () => Navigator.pop(context),
                  child: Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.arrow_back_rounded,
                        color: Colors.white),
                  ),
                ),
              ),
              // live badge
              Positioned(
                top: 48, right: 16,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: kRed.withValues(alpha: 0.85),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.circle, color: Colors.white, size: 8),
                      SizedBox(width: 6),
                      Text('LIVE',
                          style: TextStyle(
                              color: Colors.white,
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1)),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

// ── live overlay ─────────────────────────────────────────────

class _LiveOverlay extends StatelessWidget {
  final Map<String, dynamic> result;
  final Size capturedImageSize;

  const _LiveOverlay({
    required this.result,
    required this.capturedImageSize,
  });

  @override
  Widget build(BuildContext context) {
    final detected = result['detected'] as bool? ?? false;
    final riskLevel = result['risk_level'] as String? ?? 'Unknown';
    final confidence = (result['confidence'] as num?)?.toDouble() ?? 0;

    Color color;
    switch (riskLevel) {
      case 'High':   color = kRed; break;
      case 'Medium': color = kOrange; break;
      case 'Low':    color = kGreen; break;
      default:       color = Colors.grey;
    }

    return Stack(
      children: [
        // bbox drawn using actual image dimensions
        if (detected && result['bbox'] != null)
          CustomPaint(
            size: Size.infinite,
            painter: _BBoxPainter(
              bbox: (result['bbox'] as List)
                  .map((v) => (v as num).toDouble())
                  .toList(),
              color: color,
              imageSize: capturedImageSize,
            ),
          ),

        // bottom label
        Positioned(
          bottom: 40, left: 20, right: 20,
          child: Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 20, vertical: 14),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.72),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: color.withValues(alpha: 0.5)),
            ),
            child: detected
                ? Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Risk Level',
                              style: TextStyle(
                                  color: Colors.white
                                      .withValues(alpha: 0.5),
                                  fontSize: 11)),
                          Text(riskLevel,
                              style: TextStyle(
                                  color: color,
                                  fontSize: 22,
                                  fontWeight: FontWeight.w800)),
                        ],
                      ),
                      Text(
                        '${(confidence * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold),
                      ),
                    ],
                  )
                : const Center(
                    child: Text('No dog detected',
                        style:
                            TextStyle(color: Colors.white60))),
          ),
        ),
      ],
    );
  }
}
// ── bbox painter ─────────────────────────────────────────────


class _BBoxPainter extends CustomPainter {
  final List<double> bbox;
  final Color color;
  final Size imageSize;

  _BBoxPainter({
    required this.bbox,
    required this.color,
    required this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    final rect = Rect.fromLTRB(
      bbox[0] * scaleX,
      bbox[1] * scaleY,
      bbox[2] * scaleX,
      bbox[3] * scaleY,
    );

    // box outline
    canvas.drawRect(
      rect,
      Paint()
        ..color = color.withValues(alpha: 0.85)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5,
    );

    // corner accents
    final cp = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    const len = 18.0;
    canvas.drawLine(rect.topLeft, rect.topLeft + const Offset(len, 0), cp);
    canvas.drawLine(rect.topLeft, rect.topLeft + const Offset(0, len), cp);
    canvas.drawLine(rect.topRight, rect.topRight + const Offset(-len, 0), cp);
    canvas.drawLine(rect.topRight, rect.topRight + const Offset(0, len), cp);
    canvas.drawLine(rect.bottomLeft, rect.bottomLeft + const Offset(len, 0), cp);
    canvas.drawLine(rect.bottomLeft, rect.bottomLeft + const Offset(0, -len), cp);
    canvas.drawLine(rect.bottomRight, rect.bottomRight + const Offset(-len, 0), cp);
    canvas.drawLine(rect.bottomRight, rect.bottomRight + const Offset(0, -len), cp);

    // label above box
    final textPainter = TextPainter(
      text: TextSpan(
        text: '  Dog  ',
        style: TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
          backgroundColor: color,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    textPainter.paint(canvas, Offset(rect.left, rect.top - 18));
  }

  @override
  bool shouldRepaint(_BBoxPainter old) =>
      old.bbox != bbox || old.imageSize != imageSize;
}



// ============================================================
// BBOX IMAGE WIDGET (for static image result)
// ============================================================

class _BboxImageWidget extends StatefulWidget {
  final Uint8List imageBytes;
  final List<double> bbox;
  final String riskLevel;

  const _BboxImageWidget({
    required this.imageBytes,
    required this.bbox,
    required this.riskLevel,
  });

  @override
  State<_BboxImageWidget> createState() => _BboxImageWidgetState();
}

class _BboxImageWidgetState extends State<_BboxImageWidget> {
  ui.Image? _uiImage;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final codec = await ui.instantiateImageCodec(widget.imageBytes);
    final frame = await codec.getNextFrame();
    if (mounted) setState(() => _uiImage = frame.image);
  }

  Color get _color {
    switch (widget.riskLevel) {
      case 'High':   return kRed;
      case 'Medium': return kOrange;
      case 'Low':    return kGreen;
      default:       return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_uiImage == null) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(18),
        child: Image.memory(widget.imageBytes,
            fit: BoxFit.contain, width: double.infinity),
      );
    }

    final imageW = _uiImage!.width.toDouble();
    final imageH = _uiImage!.height.toDouble();

    return ClipRRect(
      borderRadius: BorderRadius.circular(18),
      child: LayoutBuilder(builder: (context, constraints) {
        final displayW = constraints.maxWidth;
        final displayH = displayW * imageH / imageW;

        return SizedBox(
          width: displayW,
          height: displayH,
          child: Stack(
            children: [
              Image.memory(
                widget.imageBytes,
                fit: BoxFit.contain,
                width: displayW,
                height: displayH,
              ),
              CustomPaint(
                size: Size(displayW, displayH),
                painter: _BBoxPainter(
                  bbox: widget.bbox,
                  color: _color,
                  imageSize: Size(imageW, imageH),
                ),
              ),
            ],
          ),
        );
      }),
    );
  }
} 

// ============================================================
// SETTINGS SCREEN
// ============================================================

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _ctrl = TextEditingController();
  bool _saved = false;
  bool _checking = false;
  bool? _serverOk;

  @override
  void initState() {
    super.initState();
    _ctrl.text = ApiService.baseUrl;
  }

  Future<void> _save() async {
    await ApiService.saveUrl(_ctrl.text.trim());
    setState(() => _saved = true);
    Future.delayed(const Duration(seconds: 2),
        () { if (mounted) setState(() => _saved = false); });
  }

  Future<void> _test() async {
    setState(() { _checking = true; _serverOk = null; });
    final ok = await ApiService.checkHealth();
    setState(() { _serverOk = ok; _checking = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      appBar: AppBar(
        backgroundColor: kBackground,
        title: const Text('Settings',
            style: TextStyle(fontWeight: FontWeight.bold)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Server Configuration',
                style: TextStyle(
                    fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            const Text(
              'Enter your ngrok URL to enable online mode.',
              style: TextStyle(color: Colors.white54, fontSize: 13),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _ctrl,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: 'ngrok URL',
                hintText: 'https://xxxx.ngrok-free.app',
                hintStyle: const TextStyle(color: Colors.white30),
                labelStyle: const TextStyle(color: Colors.white54),
                filled: true,
                fillColor: kCard,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(14),
                  borderSide: BorderSide.none,
                ),
                prefixIcon: const Icon(Icons.link_rounded,
                    color: kAmber),
              ),
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _save,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: kAmber,
                      foregroundColor: Colors.black,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    child: Text(_saved ? 'Saved ✓' : 'Save URL',
                        style: const TextStyle(
                            fontWeight: FontWeight.bold)),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton(
                    onPressed: _checking ? null : _test,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      side: BorderSide(
                          color: Colors.white.withValues(alpha: 0.15)),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    child: _checking
                        ? const SizedBox(
                            width: 16, height: 16,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: kAmber))
                        : Text(
                            _serverOk == null
                                ? 'Test Connection'
                                : _serverOk!
                                    ? '✓ Connected'
                                    : '✗ Failed',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: _serverOk == null
                                  ? Colors.white
                                  : _serverOk!
                                      ? kGreen
                                      : kRed,
                            ),
                          ),
                  ),
                ),
              ],
            ),
            if (_serverOk != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: (_serverOk! ? kGreen : kRed)
                      .withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                      color: (_serverOk! ? kGreen : kRed)
                          .withValues(alpha: 0.3)),
                ),
                child: Row(
                  children: [
                    Icon(
                      _serverOk!
                          ? Icons.check_circle_rounded
                          : Icons.error_rounded,
                      color: _serverOk! ? kGreen : kRed,
                      size: 18,
                    ),
                    const SizedBox(width: 10),
                    Text(
                      _serverOk!
                          ? 'Server is reachable. Online mode ready.'
                          : 'Cannot reach server. Check the URL and make sure Flask + ngrok are running.',
                      style: TextStyle(
                          color: _serverOk! ? kGreen : kRed,
                          fontSize: 13),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ============================================================
// SHARED WIDGETS
// ============================================================

class _ModeTab extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool active;
  final VoidCallback onTap;

  const _ModeTab({
    required this.label,
    required this.icon,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: active ? kAmber : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon,
                  size: 16,
                  color: active ? Colors.black : Colors.white54),
              const SizedBox(width: 6),
              Text(label,
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: active ? Colors.black : Colors.white54)),
            ],
          ),
        ),
      ),
    );
  }
}

class _ServerStatusBadge extends StatelessWidget {
  final bool online;
  const _ServerStatusBadge({required this.online});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: (online ? kGreen : kRed).withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
            color: (online ? kGreen : kRed).withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            online ? Icons.cloud_done_rounded : Icons.cloud_off_rounded,
            size: 13,
            color: online ? kGreen : kRed,
          ),
          const SizedBox(width: 5),
          Text(
            online ? 'Server Connected' : 'Server Unreachable',
            style: TextStyle(
                color: online ? kGreen : kRed,
                fontSize: 11,
                fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}

class _RiskIcon extends StatelessWidget {
  final bool aggressive;
  final Color color;
  const _RiskIcon({required this.aggressive, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 64, height: 64,
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        shape: BoxShape.circle,
      ),
      child: Icon(
        aggressive ? Icons.warning_rounded : Icons.check_circle_rounded,
        color: color, size: 36,
      ),
    );
  }
}

class _RiskLevelBadge extends StatelessWidget {
  final String riskLevel;
  const _RiskLevelBadge({required this.riskLevel});

  Color get _color {
    switch (riskLevel) {
      case 'High':   return kRed;
      case 'Medium': return kOrange;
      case 'Low':    return kGreen;
      default:       return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: 64, height: 64,
          decoration: BoxDecoration(
            color: _color.withValues(alpha: 0.12),
            shape: BoxShape.circle,
          ),
          child: Icon(
            riskLevel == 'High'
                ? Icons.warning_rounded
                : riskLevel == 'Medium'
                    ? Icons.info_rounded
                    : Icons.check_circle_rounded,
            color: _color, size: 36,
          ),
        ),
        const SizedBox(height: 10),
        Text(
          '$riskLevel Risk',
          style: TextStyle(
              color: _color,
              fontSize: 22,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.3),
        ),
      ],
    );
  }
}

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
      icon: Icon(icon, size: 17),
      label: Text(label,
          style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
      style: OutlinedButton.styleFrom(
        foregroundColor: Colors.white,
        side: BorderSide(color: Colors.white.withValues(alpha: 0.12)),
        backgroundColor: Colors.white.withValues(alpha: 0.03),
        minimumSize: const Size(0, 46),
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}

class _SelectedFile extends StatelessWidget {
  final IconData icon;
  final String name;
  const _SelectedFile({required this.icon, required this.name});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 12),
      padding: const EdgeInsets.all(11),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(11),
      ),
      child: Row(
        children: [
          const Icon(Icons.check_circle_rounded,
              color: kGreen, size: 16),
          const SizedBox(width: 8),
          Icon(icon, size: 16, color: Colors.white38),
          const SizedBox(width: 8),
          Expanded(
            child: Text(name,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                    color: Colors.white60, fontSize: 12)),
          ),
        ],
      ),
    );
  }
}

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
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(number,
                  style: const TextStyle(
                      color: kAmber,
                      fontWeight: FontWeight.w800,
                      fontSize: 11)),
              const Spacer(),
              Icon(icon, color: kAmber, size: 20),
            ],
          ),
          const SizedBox(height: 12),
          Text(title,
              style: const TextStyle(
                  fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          Text(description,
              style: const TextStyle(
                  color: Colors.white54, height: 1.4, fontSize: 12)),
        ],
      ),
    );
  }
}

class _DarkInfoRow extends StatelessWidget {
  final String label;
  final String value;
  const _DarkInfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 150,
            child: Text(label,
                style: const TextStyle(
                    color: Colors.white54, fontSize: 13)),
          ),
          Expanded(
            child: Text(value,
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }
}

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
      height: 52,
      child: ElevatedButton.icon(
        onPressed: onTap,
        icon: Icon(icon),
        label: Text(title,
            style: const TextStyle(
                fontSize: 15, fontWeight: FontWeight.w700)),
        style: ElevatedButton.styleFrom(
          backgroundColor: kAmber,
          foregroundColor: Colors.black,
          elevation: 0,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(13)),
        ),
      ),
    );
  }
}