import 'package:flutter_test/flutter_test.dart';
import 'package:caninecue/main.dart';

void main() {
  testWidgets('CanineCue home screen test', (WidgetTester tester) async {
    await tester.pumpWidget(const CanineCueApp());

    expect(find.text('CanineCue'), findsOneWidget);
    expect(find.text('Dog Aggression Detection'), findsOneWidget);
  });
}