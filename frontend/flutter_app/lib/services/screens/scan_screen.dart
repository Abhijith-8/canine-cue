import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../api_service.dart';

class ScanScreen extends StatefulWidget {
  final CameraDescription camera;
  const ScanScreen({super.key, required this.camera});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  late CameraController _controller;
  late Future<void> _initFuture;

  PredictionResult? _result;
  bool _isLoading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.medium);
    _initFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _capture() async {
    setState(() { _isLoading = true; _error = null; });

    try {
      await _initFuture;
      final xFile = await _controller.takePicture();
      final imageFile = File(xFile.path);

      final result = await ApiService.predict(imageFile);
      setState(() { _result = result; _isLoading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _isLoading = false; });
    }
  }

  Color _riskColor(String level) {
    switch (level) {
      case 'High':   return Colors.red;
      case 'Medium': return Colors.orange;
      case 'Low':    return Colors.green;
      default:       return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('CanineCue Scan')),
      body: Column(
        children: [
          Expanded(
            flex: 3,
            child: FutureBuilder<void>(
              future: _initFuture,
              builder: (ctx, snap) {
                if (snap.connectionState != ConnectionState.done) {
                  return const Center(child: CircularProgressIndicator());
                }
                return CameraPreview(_controller);
              },
            ),
          ),

          // Result card
          Expanded(
            flex: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                  ? Text('Error: $_error', style: const TextStyle(color: Colors.red))
                  : _result == null
                    ? const Text('Point camera at a dog and tap Scan',
                        textAlign: TextAlign.center)
                    : _buildResultCard(_result!),
            ),
          ),

          // Scan button
          Padding(
            padding: const EdgeInsets.only(bottom: 24),
            child: ElevatedButton.icon(
              onPressed: _isLoading ? null : _capture,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Scan Dog'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard(PredictionResult r) {
    if (!r.detected) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('No dog detected. Try again.',
              textAlign: TextAlign.center, style: TextStyle(fontSize: 18)),
        ),
      );
    }

    final color = _riskColor(r.riskLevel);
    return Card(
      color: color.withValues(alpha: 0.1),
      shape: RoundedRectangleBorder(
        side: BorderSide(color: color, width: 2),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Risk Level', style: TextStyle(fontSize: 14, color: Colors.grey[600])),
            const SizedBox(height: 4),
            Text(
              r.riskLevel,
              style: TextStyle(fontSize: 36, fontWeight: FontWeight.bold, color: color),
            ),
            const SizedBox(height: 8),
            Text('Confidence: ${(r.confidence * 100).toStringAsFixed(1)}%'),
            Text('Aggression prob: ${(r.probAggressive * 100).toStringAsFixed(1)}%'),
          ],
        ),
      ),
    );
  }
}