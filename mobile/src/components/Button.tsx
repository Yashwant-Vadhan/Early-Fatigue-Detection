import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Colors } from '../constants/colors';

// 1. Define the interface for your props
interface PrimaryButtonProps {
  title: string;
  onPress: () => void;
}

// 2. Apply the interface to the props argument
export const PrimaryButton = ({ title, onPress }: PrimaryButtonProps) => (
  <TouchableOpacity style={styles.button} onPress={onPress}>
    <Text style={styles.text}>{title}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  button: {
    backgroundColor: Colors.accentCyan,
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  text: {
    color: Colors.primaryBg,
    fontWeight: 'bold',
  },
});