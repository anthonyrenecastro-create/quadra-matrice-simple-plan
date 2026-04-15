#!/bin/bash
# CognitionSim - Quick Launch Scripts

echo "🌟 CognitionSim - Launch Options"
echo "======================================"
echo ""
echo "Choose how to launch:"
echo ""
echo "1. Interactive Menu (recommended)"
echo "   python launch.py"
echo ""
echo "2. Quick Demo (20 batches, ~2 minutes)"
echo "   python -c \"import asyncio; from train_atlantean_hrm import CognitionSimTrainer; asyncio.run(CognitionSimTrainer(100).train('wikitext', 'wikitext-2-raw-v1', 20, 5))\""
echo ""
echo "3. Full Training (100 batches, ~10 minutes)"
echo "   python train_atlantean_hrm.py"
echo ""
echo "4. View Results"
echo "   python results_summary.py"
echo ""
echo "5. Help & Documentation"
echo "   cat TRAINING.md"
echo ""
echo "======================================"
echo ""
read -p "Enter number to launch (1-5) or q to quit: " choice

case $choice in
    1)
        python launch.py
        ;;
    2)
        echo "🚀 Starting quick demo..."
        python -c "import asyncio; from train_atlantean_hrm import CognitionSimTrainer; asyncio.run(CognitionSimTrainer(100, 'cpu').train('wikitext', 'wikitext-2-raw-v1', 20, 5))"
        ;;
    3)
        echo "📊 Starting full training..."
        python train_atlantean_hrm.py
        ;;
    4)
        python results_summary.py
        ;;
    5)
        cat TRAINING.md
        ;;
    q|Q)
        echo "👋 Goodbye!"
        ;;
    *)
        echo "❌ Invalid choice. Run './start.sh' again."
        ;;
esac
