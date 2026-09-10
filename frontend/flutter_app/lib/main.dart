import 'package:flutter/material.dart';

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

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FB),

      // ---------------- APP BAR ----------------
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

      // ---------------- BODY ----------------
      body: SingleChildScrollView(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(
              maxWidth: 1100,
            ),
            child: Padding(
              padding: const EdgeInsets.all(30),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [

                  // ----------- WELCOME SECTION -----------
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
                    'Detect and analyze aggressive behavior in dogs using '
                    'AI-powered computer vision.',
                    style: TextStyle(
                      fontSize: 17,
                      color: Colors.grey,
                    ),
                  ),

                  const SizedBox(height: 35),

                  // ----------- MAIN DETECTION CARD -----------
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(30),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          blurRadius: 15,
                          spreadRadius: 2,
                          color: Colors.black.withOpacity(0.06),
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
                          'Choose an option below to analyze dog behavior.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 15,
                            color: Colors.grey,
                          ),
                        ),

                        const SizedBox(height: 30),

                        // -------- CAMERA BUTTON --------
                        SizedBox(
                          width: 260,
                          height: 55,
                          child: ElevatedButton.icon(
                            onPressed: () {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text(
                                    'Camera detection will be connected next.',
                                  ),
                                ),
                              );
                            },
                            icon: const Icon(Icons.camera_alt),
                            label: const Text(
                              'Camera Detection',
                              style: TextStyle(fontSize: 16),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.indigo,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 15),

                        // -------- UPLOAD BUTTON --------
                        SizedBox(
                          width: 260,
                          height: 55,
                          child: OutlinedButton.icon(
                            onPressed: () {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text(
                                    'Image upload will be connected next.',
                                  ),
                                ),
                              );
                            },
                            icon: const Icon(Icons.upload_file),
                            label: const Text(
                              'Upload Image',
                              style: TextStyle(fontSize: 16),
                            ),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: Colors.indigo,
                              side: const BorderSide(
                                color: Colors.indigo,
                              ),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 30),

                  // ----------- INFORMATION CARDS -----------
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [

                      // SYSTEM STATUS
                      Expanded(
                        child: _InfoCard(
                          icon: Icons.analytics,
                          title: 'Detection Status',
                          value: 'Ready',
                          description:
                              'The detection system is ready for analysis.',
                        ),
                      ),

                      const SizedBox(width: 20),

                      // AI MODEL
                      Expanded(
                        child: _InfoCard(
                          icon: Icons.smart_toy,
                          title: 'AI Model',
                          value: 'CanineCue Model',
                          description:
                              'Computer vision model for dog behavior analysis.',
                        ),
                      ),

                      const SizedBox(width: 20),

                      // SAFETY
                      Expanded(
                        child: _InfoCard(
                          icon: Icons.shield,
                          title: 'Purpose',
                          value: 'Safety',
                          description:
                              'Helps identify potentially aggressive behavior.',
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 40),

                  // ----------- ABOUT SECTION -----------
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(25),
                    decoration: BoxDecoration(
                      color: Colors.indigo,
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
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
                          'CanineCue is a dog aggression detection system '
                          'designed to analyze visual behavior and identify '
                          'potentially aggressive actions using artificial '
                          'intelligence and computer vision.',
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

                  // ----------- FOOTER -----------
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

// =====================================================
// INFORMATION CARD
// =====================================================

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
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            blurRadius: 12,
            color: Colors.black.withOpacity(0.05),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
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