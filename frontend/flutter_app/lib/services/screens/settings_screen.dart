import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _controller = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('ngrok_url') ?? ApiService.baseUrl;
    if (!mounted) return;
    setState(() {
      _controller.text = saved;
    });
  }

  Future<void> _save() async {
    final url = _controller.text.trim();
    ApiService.baseUrl = url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('ngrok_url', url);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('URL saved!')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(
                labelText: 'ngrok URL',
                hintText: 'https://xxxx.ngrok.io',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _save, child: const Text('Save')),
          ],
        ),
      ),
    );
  }
}