import React from 'react';
import { View, StyleSheet } from 'react-native';
import { PrimaryButton } from '../components/Button';
import { Colors } from '../constants/colors';

export default function Login() {
  return (
    <View style={styles.container}>
      <PrimaryButton title="Login" onPress={() => console.log('Login pressed')} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.primaryBg, justifyContent: 'center', padding: 20 }
});
